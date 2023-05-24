import pytorch_lightning as pl
import torch
from models.TFN_density import TFN_grid_density
from models.layers import MLP, MLP_layer
from utils.group_points import GroupPoints
from spherical_harmonics.spherical_cnn import torch_fibonnacci_sphere_sampling, SphericalHarmonicsEval, SphericalHarmonicsCoeffs, zernike_monoms, torch_zernike_monoms
from spherical_harmonics.kernels import SphericalHarmonicsGaussianKernels, ShGaussianKernelConv
from models.layers import MLP, MLP_layer, set_sphere_weights, apply_layers, type_1
from utils.pooling import kd_pooling_3d



class Cafi_model(torch.nn.Module):

    def __init__(self, num_capsules = 10, num_frames = 1, sphere_samples = 64, bn_momentum = 0.75, mlp_units = [[32, 32], [64, 64], [128, 256]], radius = [0.4, 0.8, 1.5]):
        super(Cafi_model, self).__init__()

        self.radius = radius
        self.bn_momentum = 0.75
        self.basis_dim = 3
        self.l_max = [3, 3, 3]
        self.l_max_out = [3, 3, 3]
        self.num_shells = [3, 3, 3]

        self.num_capsules = num_capsules
        self.num_frames = num_frames
        self.mlp_units = mlp_units
        self.TFN_arch = TFN_grid_density(sphere_samples = sphere_samples, bn_momentum = bn_momentum, mlp_units = [[32, 32], [64, 64], [128, 256]], l_max = self.l_max, l_max_out = self.l_max_out, num_shells = self.num_shells, radius = self.radius)
        self.S2 = torch_fibonnacci_sphere_sampling(sphere_samples)

        self.basis_mlp = []
        self.basis_layer = []

        self.basis_units = [64]
        for frame_num in range(num_frames):
            self.basis_mlp.append(MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.basis_units, bn_momentum = self.bn_momentum))
            self.basis_layer.append(MLP(in_channels = self.basis_units[-1], out_channels=self.basis_dim, apply_norm = False))

        self.basis_mlp = torch.nn.Sequential(*self.basis_mlp)
        self.basis_layer = torch.nn.Sequential(*self.basis_layer)


        self.code_dim = 64
        self.code_layer_params = [128]
        self.code_mlp = MLP_layer(in_channels = self.mlp_units[-1][-1], units = self.code_layer_params, bn_momentum = self.bn_momentum)
        self.code_layer = MLP(in_channels = self.code_layer_params[-1], out_channels=self.code_dim, apply_norm = False)

        self.points_inv_layer = MLP(in_channels = 128, out_channels=3, apply_norm = False)
        self.num_frames = num_frames
        self.zernike_harmonics = torch_zernike_monoms(self.l_max_out[-1])
        self.fSHT = SphericalHarmonicsCoeffs(l_max=self.l_max_out[-1], base=self.S2)
        self.type_1_basis = SphericalHarmonicsCoeffs(l_list=[1], base=self.S2)


    def forward(self, x, x_density):
        """
        x - B, N, 3 - Batch of point clouds that are kdtree indexed for pooling
        """
        
        # Compute TFN features
        F = self.TFN_arch(x, x_density)
        x_density_in = x_density        
        
        # Equivariant Basis
        E = []


        # Compute equivariant layers
       
        for frame_num in range(self.num_frames):
            basis = self.basis_mlp[frame_num](F)
            basis = self.basis_layer[frame_num](basis)
            basis = self.type_1_basis.compute(basis)["1"]
            basis = torch.nn.functional.normalize(basis, dim=-1, p = 2, eps = 1e-6)
            E.append(basis)

        B, H, W, D, _ = x.shape
        x = x.reshape(B, -1, 3)
        x_density = x_density.reshape(B, -1, 1)

        latent_code = self.code_mlp(F)
        latent_code = self.code_layer(latent_code)
        latent_code = self.fSHT.compute(latent_code)

        z = self.zernike_harmonics.compute(x)

        points_code = []

        points_inv = None
        for l in latent_code:
            p = torch.einsum('bmi,bvmj->bvij', latent_code[l], z[int(l)])
            shape = list(p.shape)
            shape = shape[:-1]
            shape[-1] = -1
            p = torch.reshape(p, shape)
            points_code.append(p)
            if int(l) == 1:
                points_inv = p

        points_code = torch.cat(points_code, dim=-1)
        points_inv = self.points_inv_layer(points_inv)

        if len(x_density_in.shape) == 4:
            x_density_in = x_density_in.unsqueeze(-1)
        
        coords = points_inv.reshape(B, H, W, D, 3)
        points_inv = torch.nn.functional.grid_sample(x_density_in.permute(0, -1, 1, 2, 3), coords, align_corners=True).squeeze(1)


        out = {"points_inv": points_inv, "E": E, "coords": coords}

        return out

