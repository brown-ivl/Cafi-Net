

import sys
sys.path.append("../")
import os
import json
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import random, sys
from scipy.ndimage import zoom
import utils.pointcloud_utils as pcd_utils
from scipy.spatial.transform import Rotation
from utils.pointcloud_utils import get_xyz_grid
import open3d as o3d
from random import randrange




class Density_loader_shapenet(Dataset):

    def __init__(self, dataset_path,sigmas_file_pattern):
        super(Density_loader_shapenet, self).__init__()
        self.dataset_path = dataset_path
        self.files = glob.glob(os.path.join(dataset_path, "") + sigmas_file_pattern)
        self.files.sort()
        self.dimensions = [32, 32, 32]
        
        
    def __len__(self):
        
        return len(self.files)
    
    def __getitem__(self, key):
    
        grid = np.load(self.files[key])
        grid = np.where(grid <=0,0,grid)
        grid = grid * grid 
        grid = np.exp(-1.0 * grid)

        grid_new = 1 - grid
        grid_new = np.clip(grid_new, 0, 1)
        density_grid = torch.from_numpy(grid_new).to(torch.float32).permute(2,1,0)
        
        pair_idx = randrange(int(len(self.files)))

        grid_1 = np.load(self.files[pair_idx])
        grid_1 = np.where(grid_1 <=0,0,grid_1)
        grid_1 = grid_1 * grid_1 
        grid_1 = np.exp(-1.0 * grid_1)

        grid_new_1= 1 - grid_1
        grid_new_1 = np.clip(grid_new_1, 0, 1)
        density_grid_1 = torch.from_numpy(grid_new_1).to(torch.float32).permute(2,1,0)
        
        density_list = [density_grid,density_grid_1]
        
        x = np.linspace(-1,1,self.dimensions[0])
        y = np.linspace(-1,1,self.dimensions[1])
        z = np.linspace(-1,1,self.dimensions[2])
        
        x_coords, y_coords,z_coords =  np.meshgrid(x,y,z,indexing='ij')
        
        coords = torch.from_numpy(np.stack((x_coords,y_coords,z_coords),axis=-1)).to(torch.float32).permute(2,1,0,3)
        coords_list =  [coords,coords]
        
        return {"density": density_list , "coords":coords_list , "file_path": self.files[key]}