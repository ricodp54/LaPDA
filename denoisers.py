import torch
from torch import nn
from models.vm import VelocityModule
from models.similarity import SimilarityModule


class DenoiserA(nn.Module):
    def __init__(self, velocity_nets=None, args=None):
        super().__init__()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn
        self.tot_its = args.tot_its
        self.num_train_points = args.num_train_points

        self.num_modules = 4


        self.velocity_nets = nn.ModuleList()
        for i in range(self.num_modules):
            if i == 0:
                self.velocity_nets.append(SimilarityModule(args=args).to(args.device))
            else:
                self.velocity_nets.append(VelocityModule(args=args).to(args.device))


    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        with torch.no_grad():
            pcl_next = pcl_noisy.clone()

            for it in range(self.tot_its):
                # Trajectories
                self.velocity_nets.eval()

                for mod in range(self.num_modules):

                    if mod == 0:
                        feat = self.velocity_nets[mod].encoder(pcl_next)
                        feat_adapt = feat + self.velocity_nets[mod].adapt_feature_noise

                        delta_f_1 = self.velocity_nets[mod].cross_attention_1(feat, feat_adapt, feat_adapt)

                        delta_f_2 = self.velocity_nets[mod].cross_attention_2(delta_f_1, feat, feat)


                        combined_features = self.velocity_nets[mod].layer_norm(delta_f_1 + delta_f_2)

                        combined_features = combined_features + self.velocity_nets[mod].mlp(combined_features)

                        F = combined_features.size(2)

                        adapt_noise = self.velocity_nets[mod].decoder(c=combined_features.view(-1, F)).reshape(B, N, d)
                        pcl_next = pcl_next + (1 / self.tot_its) * (1 / self.num_modules) * adapt_noise

                    else:
                        feat = self.velocity_nets[mod].encoder(pcl_next)
                        F = feat.size(2)

                        pred_dir = self.velocity_nets[mod].decoder(c=feat.view(-1, F)).reshape(B, N, d)

                        pcl_next = pcl_next + (1 / self.tot_its) * (1 / self.num_modules) * pred_dir  # 0.75

        return pcl_next, None





