import sys

import torch
from utils.group_points import GroupPoints
from spherical_harmonics.spherical_cnn import torch_fibonnacci_sphere_sampling, SphericalHarmonicsEval, SphericalHarmonicsCoeffs
from spherical_harmonics.kernels import SphericalHarmonicsGaussianKernels, ShGaussianKernelConv
from spherical_harmonics.kernels_density import ShGaussianKernelConv_grid, SphericalHarmonicsGaussianKernels_density
from models.layers import MLP, MLP_layer, set_sphere_weights, apply_layers
from utils.pooling import kd_pooling_3d
from utils.pointcloud_utils import GroupPoints_density, GroupPoints_grid ,GroupPoints_euclidean_density, rotate_density, get_gradient_density,kron
from spherical_harmonics.wigner_matrix import *
from e3nn import o3
from utils.train_utils import mean_center
from spherical_harmonics.clebsch_gordan_decomposition import torch_clebsch_gordan_decomposition
import open3d as o3d




class TFN_grid_density(torch.nn.Module):
    """
    TFN layer for prediction in Pytorch
    """
    def __init__(self, sphere_samples = 64, bn_momentum = 0.75, mlp_units = [[32, 32], [64, 64], [128, 256]], l_max = [3, 3, 3], l_max_out = [3, 3, 3], num_shells = [3, 3, 3], radius = [0.4]):
        super(TFN_grid_density, self).__init__()
        self.l_max = l_max
        self.l_max_out = l_max_out
        self.num_shells = num_shells
        self.gaussian_scale = []
        for i in range(len(self.num_shells)):
            self.gaussian_scale.append(0.69314718056 * ((self.num_shells[i]) ** 2))
        
        self.radius = [0.2,0.4,0.8]
        self.bounded = [True, True, True]

        self.S2 = torch_fibonnacci_sphere_sampling(sphere_samples)

        self.factor = [2, 2, 2]
        self.patch_size = [1024,512,512]

        self.spacing = [0, 0, 0]
        self.equivariant_units = [32, 64, 128]
        self.in_equivariant_channels = [[6, 13, 12, 9], [387, 874, 1065, 966], [771, 1738, 2121, 1926]]
       
        
        self.mlp_units =  mlp_units
        self.in_mlp_units = [32, 64, 128]
        self.bn_momentum = bn_momentum

        
        self.grouping_layers_e = []
        self.kernel_layers = []
        self.conv_layers = []
        self.eval = []
        self.coeffs = []

        for i in range(len(self.radius)):
           
           
            
            gi_e = GroupPoints_euclidean_density(radius=self.radius[i],
                             patch_size_source=self.patch_size[i],
                             spacing_source=self.spacing[i])
            
            self.grouping_layers_e.append(gi_e)
            

            ki = SphericalHarmonicsGaussianKernels_density(l_max=self.l_max[i],
                                                   gaussian_scale=self.gaussian_scale[i],
                                                   num_shells=self.num_shells[i],
                                                   bound=self.bounded[i])
            ci = ShGaussianKernelConv(l_max=self.l_max[i], l_max_out=self.l_max_out[i])

            self.kernel_layers.append(ki)
            self.conv_layers.append(ci)

        self.conv_layers = torch.nn.Sequential(*self.conv_layers)
        self.kernel_layers = torch.nn.Sequential(*self.kernel_layers)
        self.mlp = []
        self.equivariant_weights = []
        self.bn = []

        for i in range(len(self.radius)):
            self.bn.append(torch.nn.BatchNorm2d(self.equivariant_units[i], momentum=self.bn_momentum))
            types = [str(l) for l in range(self.l_max_out[i] + 1)]
            self.equivariant_weights.append(set_sphere_weights(self.in_equivariant_channels[i], self.equivariant_units[i], types=types))
            self.mlp.append(MLP_layer(self.in_mlp_units[i], self.mlp_units[i], bn_momentum = self.bn_momentum))

        self.mlp = torch.nn.Sequential(*self.mlp)
        self.bn = torch.nn.Sequential(*self.bn)
        self.equivariant_weights = torch.nn.Sequential(*self.equivariant_weights)

        self.iSHT = []
        self.fSHT = []
        for i in range(len(self.l_max_out)):
            self.iSHT.append(SphericalHarmonicsEval(l_max=self.l_max_out[i], base=self.S2))
            self.fSHT.append(SphericalHarmonicsCoeffs(l_max=self.l_max_out[i], base=self.S2))
            
        self.Q = torch_clebsch_gordan_decomposition(l_max=l_max_out[-1],
                                         sparse=False,
                                         output_type='dict',
                                         l_max_out=l_max_out[-1])
        
    def forward(self, x_grid, x_density):
        """
        Input:
            x_grid - [B, H, W, D, 3] - Equivariant density field
            x_density - [B, H, W, D] - Equivariant density field

        Returns:
            TFN features - F
        """
        
        if len(x_density.shape) == 4:
            x_density = x_density.unsqueeze(-1) # B, H, W, D, 1
        points_ = [x_grid]
        density_ = [x_density]
        grouped_points = []
        kernels = []

        

        for i in range(len(self.radius)):
            pi = kd_pooling_3d(points_[-1], self.factor)
            di = kd_pooling_3d(density_[-1], self.factor,'max')
            points_.append(pi)
            density_.append(di)

        points = []
        density = []
        for i in range(len(points_)):
           
            pi = points_[i]
            di = density_[i]            
            B, H, W, D, _ = pi.shape
            points.append(pi.reshape(B, -1, 3))
            density.append(di.reshape(B, -1, 1)) # B, N, 1

        yzx = []
        for i in range(len(points)):
            yzx_i = torch.stack([points[i][..., 1], points[i][..., 2], points[i][..., 0]], dim=-1)
            yzx.append(yzx_i.unsqueeze(-1)) # B, N, 3, 1
        weighted_density = []
        for i in range(len(self.radius)):
            
           
            gi = self.grouping_layers_e[i]({"source points": points[i], "target points": points[i + 1], "source density": density[i], "target density": density[i + 1]})
            weighted_density.append(gi['weighted density'])
            B, H, W, D, K, _ = gi["patches source"].shape

            
            ki = self.kernel_layers[i]({"patches": gi["patches source"].reshape(B, H*W*D, K, 3), "patches dist": gi["patches dist source"].reshape(B, H*W*D, K)})
            

            # Storing outputs
            grouped_points.append(gi)
            kernels.append(ki)
            
        ki = kernels[0]
        features = {}
        types = [0,1,2,3]
        ki = ki.squeeze(0)
        ki = ki.squeeze(-1)
        dim_start = 0
       
        

        ones_singal  = density[0].reshape(-1)
        y = {'0': ones_singal.reshape(points[0].shape[0], points[0].shape[1], 1, 1).type_as(x_grid)}
        for i in range(len(self.radius)):
            y["source points"] = points[i]
            y["target points"] = points[i + 1]
            B, H, W, D, K, _ = grouped_points[i]["patches idx source"].shape
            y["patches idx"] = grouped_points[i]["patches idx source"].reshape(B, H*W*D, K, -1)
            y["patches dist source"] = grouped_points[i]["patches dist source"].reshape(B, H*W*D, K)
            y["kernels"] = kernels[i]

            if "patches mask" in grouped_points[i]:
                y["patches mask"] = grouped_points[i]["patches mask"]
            
            
            shape_ = density_[i].shape
         
            gradient_density_signal_int = get_gradient_density(density[i].reshape(shape_[0],shape_[1],shape_[2],shape_[3]))
            gradient_density_signal     = gradient_density_signal_int.permute(0,2,3,4,1).reshape(shape_[0],-1,3,1)
            
            gradeint_density_signal_x  =  get_gradient_density( gradient_density_signal_int[:,0,...]).permute(0,2,3,4,1).reshape(shape_[0],-1,3,1)
            
           
            if '1' in y:
                y['1'] = torch.cat([y['1'], gradient_density_signal], dim=-1)
            else:
                y['1'] =  gradient_density_signal
            
                
         
            y = self.conv_layers[i](y)
            
            #return y
            shape_ = density_[i+1].shape
            
            gradient_density_signal_int_1 = get_gradient_density(density[i+1].reshape(shape_[0],shape_[1],shape_[2],shape_[3]))
            gradient_density_signal_1       = gradient_density_signal_int_1.permute(0,2,3,4,1).reshape(shape_[0],-1,3,1)
            
            
            
            if '1' in y:
                y['1'] = torch.cat([y['1'],  gradient_density_signal_1], dim=-1)
            else:
                y['1'] =  gradient_density_signal_1
            
                
          
            
            
            for key in y.keys():
                if key.isnumeric():
                    y[key] = y[key]* density[i+1].unsqueeze(-1)
            
            
          

            y = apply_layers(y, self.equivariant_weights[i]) # B, d, 2*l + 1, C
            

           
            y = self.iSHT[i].compute(y)
            y = y.permute(0, 3, 1, 2)
            y = self.bn[i](y)
            y = torch.nn.ReLU(True)(y)
            y = y.permute(0, 2, 3, 1)
            y = self.mlp[i](y)
            
            if i < len(self.radius) - 1:
                # Spherical Harmonic Transform
                y = self.fSHT[i].compute(y)
               
       
        
        F = torch.max(y, dim=1, keepdims=False).values # B, samples, feature_dim

        return F