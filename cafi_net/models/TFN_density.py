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

        self.grouping_layers = []
        self.grouping_layers_e = []
        self.kernel_layers = []
        self.conv_layers = []
        self.eval = []
        self.coeffs = []

        for i in range(len(self.radius)):
            gi = GroupPoints_density(radius=self.radius[i],
                             patch_size_source=self.patch_size[i],
                             spacing_source=self.spacing[i])
            
            gi_e = GroupPoints_euclidean_density(radius=self.radius[i],
                             patch_size_source=self.patch_size[i],
                             spacing_source=self.spacing[i])
            self.grouping_layers.append(gi)
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
            
            # Finding nearest neighbors of each point to compute graph features
            
            
            gi = self.grouping_layers_e[i]({"source points": points[i], "target points": points[i + 1], "source density": density[i], "target density": density[i + 1]})
            weighted_density.append(gi['weighted density'])
            B, H, W, D, K, _ = gi["patches source"].shape

            # Computing kernels for patch neighbors
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
        
        
        
        
        
        ones_singal =torch.ones((points[0].shape[0], points[0].shape[1], 1, 1)).type_as(x_grid).reshape(-1)
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
            
            
            if '1' in y:
                y['1'] = torch.cat([y['1'], gradient_density_signal], dim=-1)
            else:
                y['1'] =  gradient_density_signal
                 
            
            y = self.conv_layers[i](y)
            
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
                y = self.fSHT[i].compute(y)
        
        
        F = torch.max(y, dim=1, keepdims=False).values # B, samples, feature_dim

        return F


if __name__ == "__main__":
    sdfPath = "/home2/aditya.sharm/brics_fields/res_64/plane/train/rotated_fields/02691156_807d735fb9860fd1c863ab010b80d9ed_64_8_sdf.npy"
    ptsPath = "/home2/aditya.sharm/brics_fields/res_64/plane/train/rotated_fields/02691156_807d735fb9860fd1c863ab010b80d9ed_64_8_pts.npy"

    sdf = np.load(sdfPath,allow_pickle=False)
    coords = np.load(ptsPath,allow_pickle=False)
    
    scale_factor = (coords.max())
    coords = coords / scale_factor
    
    x_in = torch.from_numpy(coords)
    x_density = torch.from_numpy(sdf)
    
    x = x_in
    
    
    
    x_density = x_density.to(torch.float64)
    model = TFN_grid_density()
    
    type_features = model(x.unsqueeze(0).to(torch.float32), x_density.unsqueeze(0).to(torch.float32))
    
    print(type_features.shape) 
    euler_angles_tensor = torch.tensor([2.0*0.785398,0,0])
    
    rot_mat = euler_rot_zyz(2.0*0.785398,0,0)
    types= [1,2,3]
    wigner_rot_dict = {}
    print(rot_mat)
    
    
    for type_ in types:
        
        wigner_rot = o3.wigner_D(type_, euler_angles_tensor[0], euler_angles_tensor[1], euler_angles_tensor[2])
        wigner_rot = wigner_rot
        print(wigner_rot)
        wigner_rot_dict[str(type_)] = wigner_rot.to(torch.float64)
        
    
    rotated_features_dict = {}
    
    for type_ in types:
        rot_features = torch.matmul(wigner_rot_dict[str(type_)].to(torch.float32),type_features[str(type_)])
        rotated_features_dict[str(type_)] = rot_features
    
    
    rot_mat = torch.from_numpy(rot_mat)
    rot_mat = rot_mat
    rot_mat = rot_mat.type(torch.float32)
    rot_mat = rot_mat.type_as(x_density)
    
    rot_mat = rot_mat.unsqueeze(0)
    
    un_rot_density_zyx = x_density.unsqueeze(0).unsqueeze(0)
    un_rot_density_zyx = un_rot_density_zyx.permute(0,1,4,3,2)
    
    x_density_rot_zyx = rotate_density(torch.inverse(rot_mat), un_rot_density_zyx)
    
    x_density_rot_zyx = x_density_rot_zyx.squeeze(0)
    x_density_rot_zyx = x_density_rot_zyx.squeeze(0)
    x_density_rot = x_density_rot_zyx.permute(2,1,0)
    
    x_density_rot[x_density_rot >= 0.5] = 1
    x_density_rot[x_density_rot < 0.5]  = 0
    

     
    rot_type_features = model(x.unsqueeze(0).to(torch.float32), x_density_rot.unsqueeze(0).to(torch.float32))
    
    '''
    for type_ in types:
        rotated_features_dict[str(type_)] = torch.sum(rotated_features_dict[str(type_)],dim=-1)
        rot_type_features[str(type_)] = torch.sum(rot_type_features[str(type_)],dim=-1)
    '''
    
    
    
    
    main_idx = 0
    max_dist_list = []
    
    channels = [12,12,9]
    
    for type_ in types:
        
        for ch_dim in range(channels[type_-1]):
            rot_feature = rot_type_features[str(type_)][:,:,:,ch_dim].squeeze(-1)
            dim = (2*type_)+1
            rot_feature = rot_feature.reshape(16,16,16,dim).unsqueeze(0).permute(0,4,3,2,1)
            rot_feature_ = rotate_density(rot_mat, rot_feature).squeeze(0)
            rot_sigma_feature_ = rot_feature_.permute(3,2,1,0).reshape(-1,dim)
            rot_point_feature = rotated_features_dict[str(type_)][:,:,:,ch_dim].squeeze(-1).squeeze(0)
            match = 0
            un_match = 0
            non_zero_features = 0
            for index in range(rot_sigma_feature_.shape[0]):
              if torch.count_nonzero(rot_sigma_feature_[index].type(torch.LongTensor)) > 0:
                non_zero_features = non_zero_features + 1
              else:
                continue
              if  torch.allclose(rot_sigma_feature_[index],rot_point_feature[index],atol=1e-04):
                  match = match +1 
              else:
                  un_match = un_match + 1
                
            print("matched features for the type ", type_ ," = ",match)
            #print("significant features for the type ", type_ ," = ",non_zero_features)
        
    '''  
    for type_ in types:
        rot_feature = rotated_features_dict[str(type_)]
        rot_point_feature = rot_type_features[str(type_)]
        
        if  torch.allclose(rot_feature,rot_point_feature,atol=1e-03):
            print("Equal for the feature type ",str(type_))
            print("Differece max ",torch.max(torch.abs(rot_feature - rot_point_feature)))
            print("Differece sum ",torch.sum(torch.abs(rot_feature - rot_point_feature)))
            print("Differece min ",torch.min(torch.abs(rot_feature - rot_point_feature)))
            print("Differece mean ",torch.mean(torch.abs(rot_feature - rot_point_feature)))
            print("Differece median ",torch.median(torch.abs(rot_feature - rot_point_feature)))
        else:
            
            print("Not Equal for the feature type ",str(type_))
            print("Differece Norm ",torch.linalg.norm(torch.abs(rot_feature - rot_point_feature), dim=-1).max(),)
            print("Differece max ",torch.max(torch.abs(rot_feature - rot_point_feature)))
            print("Differece sum ",torch.sum(torch.abs(rot_feature - rot_point_feature)))
            print("Differece min ",torch.min(torch.abs(rot_feature - rot_point_feature)))
            print("Differece mean ",torch.mean(torch.abs(rot_feature - rot_point_feature)))
            print("Differece median ",torch.median(torch.abs(rot_feature - rot_point_feature)))
            print("Differece Mode ",torch.mode(torch.abs(rot_feature - rot_point_feature)))
    
    for type_ in types:
        rot_feature = rotated_features_dict[str(type_)][:,:,:,5:6].squeeze(-1).squeeze(0)
        rot_point_feature = rot_type_features[str(type_)]
        match = 0
        un_match = 0
        for index in range(rot_feature.shape[0]):
          
          if  torch.allclose(rot_feature[index],rot_point_feature[index],atol=1e-03):
              match = match +1 
          else:
              un_match = un_match + 1
            
        print("matched features for the type ", type_ ," = ",match)'''
    
    '''
    main_idx = 0
    for type_ in types:
        rot_feature = rotated_features_dict[str(type_)].squeeze(0)
        rot_point_feature = rot_type_features[str(type_)].squeeze(0)
        
        search_idx = []
        for main_idx in range(rot_feature.shape[0]):
            main_feature = rot_feature[:,:,0:1].squeeze(-1)[main_idx]
            found = 0
            for second_idx in range(rot_point_feature.shape[0]):
                second_feature = rot_point_feature[second_idx]
                if second_idx in search_idx:
                    continue
                if  torch.allclose(main_feature,second_feature,atol=1e-03):
                    #print("Equal for the feature type  ",str(type_)," at the idx", main_idx)
                    search_idx.append(second_idx)
                    found = 1
                    break
            if not found:
                print("Not Fo")'''
