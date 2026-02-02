import os
# import torch
# import pytorch3d
# import pytorch3d.loss
import numpy as np
# from scipy.spatial.transform import Rotation
import pandas as pd
import point_cloud_utils as pcu
from tqdm.auto import tqdm
# from triton.language import dtype
from matplotlib import colors
import matplotlib.cm as cm
from models.utils import *


def load_xyz(xyz_dir):
    all_pcls = {}
    dir_list = sorted(os.listdir(xyz_dir))
    dir_list.sort()
    for fn in tqdm(dir_list, desc='Loading'):
        if fn[-3:] != 'xyz':
            continue
        name = fn[:-4]
        path = os.path.join(xyz_dir, fn)
        all_pcls[name] = torch.FloatTensor(np.loadtxt(path, dtype=np.float32))
    return all_pcls

def load_off(off_dir):
    all_meshes = {}
    dir_list = os.listdir(off_dir)
    dir_list.sort()
    for fn in tqdm(dir_list, desc='Loading'):
        if fn[-3:] != 'off':
            continue
        name = fn[:-4]
        path = os.path.join(off_dir, fn)
        verts, faces = pcu.load_mesh_vf(path)
        verts = torch.FloatTensor(verts)
        faces = torch.LongTensor(faces)
        all_meshes[name] = {'verts': verts, 'faces': faces}
    return all_meshes


def colorize(p2s, vmin=0, vmax=1, norm='Normalize', identical_color=False):
    # cmap_name = 'coolwarm'
    # cmap = cm.get_cmap(cmap_name)  # PiYG

    vals = np.ones((256, 4))
    # min_color = np.array([225, 225, 254]) / 256  # 118, 170, 232
    # mid_color = np.array([22, 22, 135]) / 256
    # max_color = np.array([247, 239, 2]) / 256

    min_color = np.array([180, 210, 250]) / 255  # 更冷一点的浅蓝
    mid_color = np.array([0, 51, 153]) / 255  # 更深的蓝（亮度低）
    max_color = np.array([255, 215, 0]) / 255

    vals[:, 0] = np.concatenate(
        [np.linspace(min_color[0], mid_color[0], 64), np.linspace(mid_color[0], max_color[0], 192)])
    vals[:, 1] = np.concatenate(
        [np.linspace(min_color[1], mid_color[1], 64), np.linspace(mid_color[1], max_color[1], 192)])
    vals[:, 2] = np.concatenate(
        [np.linspace(min_color[2], mid_color[2], 64), np.linspace(mid_color[2], max_color[2], 192)])

    cmap = colors.ListedColormap(vals)

    if identical_color:
        norm = cm.colors.Normalize(vmin=0.0, vmax=0.0)
    elif norm == 'Normalize':
        norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
    elif norm == 'Power':
        norm = cm.colors.PowerNorm(2, vmin, vmax)
    elif norm == 'TwoSlope':
        norm = cm.colors.TwoSlopeNorm(vmin + (vmax - vmin) * 0.2, vmin, vmax)
    elif norm == 'Log':
        norm = cm.colors.LogNorm(vmin + 1e-5, vmax + 1e-5)
    else:
        assert False
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(p2s)[:, :3]

def cal_p2m(pcl,mesh_file):
    pcl_up = torch.FloatTensor(pcl)
    verts, faces = pcu.load_mesh_vf(mesh_file)

    verts = torch.FloatTensor(verts).to('cuda')
    faces = torch.LongTensor(faces).to('cuda')

    pcl_up = pcl_up.unsqueeze(0).to('cuda')

    distances = pointwise_p2m_distance_normalized(
        pcl=pcl_up[0],
        verts=verts,
        faces=faces
    ).cpu().numpy()

    p2f = point_mesh_bidir_distance_single_unit_sphere(
        pcl=pcl_up[0],
        verts=verts,
        faces=faces
    ).item()

    vmax = 16e-4
    rgb = colorize(distances, vmin=0.0, vmax=vmax, norm='TwoSlope', identical_color=False)

    return p2f*1e4,rgb