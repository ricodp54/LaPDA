import torch
from torch import nn
from torch.distributions.uniform import Uniform
from pytorch3d.ops import knn_points
import numpy as np

from models.feature import FeatureExtraction
from models.decoder import Decoder


def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]


class VelocityModule(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # geometry
        self.frame_knn = 32
        self.tot_its = args.tot_its
        # self.num_train_points = args.num_train_points
        # score-matching
        # self.dsm_sigma = args.dsm_sigma
        # networks
        self.encoder = FeatureExtraction(k=self.frame_knn, input_dim=3, embedding_dim=args.feat_embedding_dim)
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=args.decoder_hidden_dim,
        )


    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        tot_steps = self.tot_its
        with torch.no_grad():
            pcl_next = pcl_noisy.clone()

            for it in range(tot_steps):
                # Trajectories
                self.encoder.eval()
                self.decoder.eval()

                feat = self.encoder(pcl_next)
                F = feat.size(2)

                # frame_centered = pcl_next.unsqueeze(2)
                pred_dir = self.decoder(c=feat.view(-1, F)).reshape(B, N, d)
                pcl_next += (1 / tot_steps) * pred_dir

        return pcl_next, None