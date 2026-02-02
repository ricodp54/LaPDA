import torch
from torch import nn
from torch.distributions.uniform import Uniform
from pytorch3d.ops import knn_points
import numpy as np

from models.feature import FeatureExtraction
from models.decoder import Decoder
from models.cross_attention import CrossAttentionLayer
from torch.nn import Sequential as Seq, Linear, ReLU


def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]


class SimilarityModule(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn
        self.num_train_points = args.num_train_points
        # score-matching
        self.dsm_sigma = args.dsm_sigma
        # networks
        # self.encoder = FeatureExtraction(k=self.frame_knn, input_dim=3, embedding_dim=args.feat_embedding_dim)
        # self.decoder = Decoder(
        #     z_dim=self.encoder.embedding_dim,
        #     dim=3,
        #     out_dim=3,
        #     hidden_size=args.decoder_hidden_dim,
        # )

        self.hidden_dim = 256
        self.encoder = FeatureExtraction(k=self.frame_knn, input_dim=3, embedding_dim=self.hidden_dim)
        self.adapt_feature_noise = nn.Parameter(torch.zeros(size=(1, args.patch_size, self.hidden_dim)))
        self.cross_attention_1 = CrossAttentionLayer(self.hidden_dim)
        self.cross_attention_2 = CrossAttentionLayer(self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.mlp = Seq(Linear(self.hidden_dim, self.hidden_dim),
                       # BN(out_channels),
                       ReLU(),
                       Linear(self.hidden_dim, self.hidden_dim))

        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=args.decoder_hidden_dim,
        )

    def get_supervised_loss(self, pcl_noisy_L2, pcl_noisy, pcl_clean, pcl_std):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
        """
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        feat = self.encoder(pcl_noisy)
        feat_adapt = feat + self.adapt_feature_noise

        delta_f_1 = self.cross_attention_1(feat, feat_adapt, feat_adapt)
        # f_1 = feat_adapt + delta_f_1

        delta_f_2 = self.cross_attention_2(delta_f_1, feat, feat)
        # f_2 = feat + delta_f_2

        combined_features = self.layer_norm(delta_f_1 + delta_f_2)

        combined_features = combined_features + self.mlp(combined_features)

        F = combined_features.size(2)

        combined_features = combined_features[:, pnt_idx, :]
        pcl_noisy_L2 = pcl_noisy_L2[:, pnt_idx, :]
        # pcl_noisy = pcl_noisy[:, pnt_idx, :]
        pcl_clean = pcl_clean[:, pnt_idx, :]

        grad_dir_t_target = pcl_clean - pcl_noisy_L2

        adapt_noise = self.decoder(c=combined_features.view(-1, F)).reshape(B, len(pnt_idx), d)
        loss = (((adapt_noise - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()


        return loss

    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        tot_steps = 4
        with torch.no_grad():
            pcl_next = pcl_noisy.clone()

            for it in range(tot_steps):
                # Trajectories

                self.encoder.eval()
                self.cross_attention_1.eval()
                self.cross_attention_2.eval()
                self.layer_norm.eval()
                self.decoder.eval()

                feat = self.encoder(pcl_next)
                feat_adapt = feat + self.adapt_feature_noise

                delta_f_1 = self.cross_attention_1(feat, feat_adapt, feat_adapt)
                # f_1 = feat_adapt + delta_f_1

                delta_f_2 = self.cross_attention_2(delta_f_1, feat, feat)
                # f_2 = feat + delta_f_2

                combined_features = self.layer_norm(delta_f_1 + delta_f_2)

                combined_features = combined_features + self.mlp(combined_features)

                F = combined_features.size(2)

                adapt_noise = self.decoder(c=combined_features.view(-1, F)).reshape(B, N, d)
                pcl_next += (1 / tot_steps) * adapt_noise

        return pcl_next, None