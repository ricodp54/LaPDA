import argparse

import numpy as np
import torch
import torch.utils.tensorboard


from utils.misc import *
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

parser.add_argument('--ckpt', type=str, default='./pretrain/LaPDA.pt')
parser.add_argument('--input_root', type=str, default='./data/examples')
parser.add_argument('--output_root', type=str, default='./data/results')
parser.add_argument('--dataset_root', type=str, default='./data/')
parser.add_argument('--type', type=str, default='Gaussian')  # Gaussian
parser.add_argument('--dataset', type=str, default='PUNet')  # PUNet, PCNet
parser.add_argument('--resolution', type=str, default='10000_poisson')  # resolution_list = ['50000_poisson', '10000_poisson']
parser.add_argument('--noise', type=str, default='0.01')
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

# Input/Output
if args.dataset == "PUNet" or args.dataset == 'PCNet':
    input_dir = os.path.join(args.input_root, args.type,
                             '%s_%s_%s' % (args.dataset, args.resolution, args.noise))
else:
    input_dir = os.path.join(args.input_root, '%s' % (args.dataset))

save_title = '{method}_{dataset}_{res}_{noise}'.format_map({
    'method': 'LaPDA',
    'dataset': args.dataset,
    'res': args.resolution,
    'noise': args.noise
})
output_dir = os.path.join(args.output_root, args.type, 'LaPDA',save_title)
if args.dataset != "PUNet" and args.dataset != 'PCNet':
    output_dir = os.path.join(args.output_root, '%s' % (args.dataset), 'LaPDA')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Output point clouds

print('processed %s!' % output_dir)

# Denoise
start_time = time.time()
for shape_num, data in enumerate(input_iter(input_dir)):
    pcl_noisy = data['pcl_noisy'].to(args.device)
    with torch.no_grad():
        model.eval()
        pcl_next = pcl_noisy
        try:
            for niter in range(args.niters):
                pcl_next = patch_based_denoise(
                    model=model,
                    pcl_noisy=pcl_next,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    patch_size=args.patch_size,
                )

        except Exception as e:
            print("=" * 100)
            print(e)
            print("=" * 100)
            print('Current niter is {}'.format(niter))
            print("=" * 100)

        pcl_denoised = pcl_next.cpu()
        # Denormalize
        pcl_denoised = pcl_denoised * data['scale'] + data['center']

    save_path = os.path.join(output_dir, data['name'] + '.xyz')
    np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')
