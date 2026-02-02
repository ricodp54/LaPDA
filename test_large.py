import argparse

import numpy as np
import torch
import torch.utils.tensorboard


# from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from utils.evaluate import *
from denoisers import *


def input_iter(input_dir):
    for fn in sorted(os.listdir(input_dir)):
        if fn[-3:] != 'xyz':
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {
            'pcl_noisy': pcl_noisy,
            'name': fn[:-4],
            'center': center,
            'scale': scale
        }


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--ckpt', type=str, default='/pretrain/LaPDA.pt')
parser.add_argument('--input_root', type=str, default='./data/examples')
parser.add_argument('--output_root', type=str, default='./data/results')
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--type', type=str, default='')  # UnkownNoise
parser.add_argument('--dataset', type=str, default='')  # kitti_360, RueMadame
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--resolution', type=str, default='')  # raw_scan
parser.add_argument('--noise', type=str, default='')  # UnkownLevel
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2025)
# Denoiser parameters
parser.add_argument('--patch_size', type=int, default=1000)
parser.add_argument('--seed_k', type=int, default=6)
parser.add_argument('--seed_k_alpha', type=int, default=1)  # RueMadame: 10,others: 1
parser.add_argument('--niters', type=int, default=1)
args = parser.parse_args()
seed_all(args.seed)


# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = DenoiserA(args=ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

input_dir = os.path.join(args.input_root, '%s' % (args.dataset))
save_title = '{method}_{dataset}_{res}_{noise}'.format_map({
    'method': 'LapDA',
    'dataset': args.dataset,
    'res': args.resolution,
    'noise': args.noise
})
output_dir = os.path.join(args.output_root, args.type, save_title)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for name in os.listdir(input_dir):
    input_xyz = os.path.join(input_dir,name)
    output_xyz = os.path.join(output_dir,name)

    pcl= np.loadtxt(input_xyz)
    pcl = torch.FloatTensor(pcl).to('cuda')

    for iter in range(args.niters):
        pcl = denoise_large_pointcloud(model=model,
                                       pcl=pcl,
                                       cluster_size=30000,
                                       seed=args.seed)
    pcl = pcl.cpu().numpy()
    np.savetxt(output_xyz,pcl,fmt='%.8f')
