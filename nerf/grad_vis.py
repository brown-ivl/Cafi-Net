import open3d as o3d
import numpy as np
import os
import argparse

import sys
import open3d as o3d
import seaborn as sns
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import torch
from scipy.spatial import distance
import torch.nn
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter

def get_gradient_density(x):
    # x - B, H, W, D

    B, H, W, D = x.shape
    d_x = x[..., 1:, :, :] - x[..., :-1, :, :] # B, H - 1, W, D
    d_y = x[..., :, 1:, :] - x[..., :, :-1, :] # B, H, W - 1, D
    d_y = -d_y
    d_z = x[..., :, :, 1:] - x[..., :, :, :-1] # B, H, W, D - 1

    d_x = torch.nn.functional.interpolate(d_x.unsqueeze(1), (H, W, D)).squeeze(1)
    d_y = torch.nn.functional.interpolate(d_y.unsqueeze(1), x.shape[1:]).squeeze(1)
    d_z = torch.nn.functional.interpolate(d_z.unsqueeze(1), x.shape[1:]).squeeze(1)

    gradient = torch.stack([d_x, d_y, d_z], 1)

    return gradient # B, 3, H, W, D


def draw_oriented_pointcloud(x, n, t=1.0):
    a = x
    b = x + t * n
    points = []
    lines = []
    for i in range(a.shape[0]):
        ai = [a[i, 0], a[i, 1], a[i, 2]]
        bi = [b[i, 0], b[i, 1], b[i, 2]]
        points.append(ai)
        points.append(bi)
        lines.append([2*i, 2*i+1])
    colors = [[1, 0, 0] for i in range(len(lines))]

    pcd = o3d.geometry.PointCloud()
    point_colors = np.ones(x.shape)
    pcd.points = o3d.utility.Vector3dVector(a)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[-1, -1, -1])
    o3d.visualization.draw_geometries([line_set, pcd,mesh_frame])


parser = argparse.ArgumentParser(description="NeRF density gradient field visualization")
parser.add_argument("--input_dir", required=True, type = str)
parser.add_argument("--res", default=32, type = int)
parser.add_argument("--max_files", default=10, type = int)
args = parser.parse_args()

count = 0
for file in os.listdir(args.input_dir):
    if "_sigmas_%d.npy" % (args.res) not in file:
        continue

    sigmas_path = os.path.join(input_dir, file)
    samples_file = file.replace("sigmas", "samples")
    samples_path = os.path.join(input_dir, samples_file)

    density = np.load(sigmas_path)
    coords = np.load(samples_path)

    density = density.transpose(2, 1, 0)
    density = gaussian_filter(density, sigma=1)
    density = torch.from_numpy(density)
    
    coords = coords.transpose(2, 1, 0, 3)
    coords = coords.reshape(-1, 3)
    scale_ = 0.1
    
    grad = get_gradient_density(density.unsqueeze(0)).squeeze(0)
    grad = grad.permute(1, 2, 3, 0).reshape(-1,3)
    grad = grad.detach().numpy()
    draw_oriented_pointcloud(coords, grad, scale_)

    count += 1
    if count >= args.max_files:
        break
