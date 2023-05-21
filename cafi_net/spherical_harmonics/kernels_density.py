from spherical_harmonics.kernels import *
from utils.pointcloud_utils import GroupPoints_grid, GroupPoints_density

class SphericalHarmonicsGaussianKernels_density(torch.nn.Module):
    def __init__(self, l_max, gaussian_scale, num_shells, transpose=False, bound=True):
        super(SphericalHarmonicsGaussianKernels_density, self).__init__()
        self.l_max = l_max
        self.monoms_idx = torch_monomial_basis_3D_idx(l_max)
        self.gaussian_scale = gaussian_scale
        self.num_shells = num_shells
        self.transpose = True
        self.Y = torch_spherical_harmonics_basis(l_max, concat=True)
        self.split_size = []
        self.sh_idx = []
        self.bound = bound
        for l in range(l_max + 1):
            self.split_size.append(2*l+1)
            self.sh_idx += [l]*(2*l+1)
        self.sh_idx = torch.from_numpy(np.array(self.sh_idx)).to(torch.int64)



    def forward(self, x):
        #target_density = x["target_density"]
        #target_density = target_density.squeeze(-1)
        
        #target_density[target_density > 0.] = 1
        #target_density[target_density <= 0.]  = 0
        
        
        if "patches dist" in x:
            patches_dist = x["patches dist"].unsqueeze(-1)
        else:
            patches_dist = torch.linalg.norm(x["patches"], dim=-1, keepdims=True)
        normalized_patches = x["patches"] / torch.maximum(patches_dist, torch.tensor(0.000001).type_as(x["patches"]))
        if self.transpose:
            normalized_patches = -normalized_patches
        # print(normalized_patches.shape)
        monoms_patches = torch_eval_monom_basis(normalized_patches, self.l_max, idx=self.monoms_idx)
        # print(self.Y.shape)
        
        #x["patches_density"][target_density == 0, ...] = 0.
        
        #sh_patches = x["patches_density"] * torch.einsum('ij,bvpj->bvpi', self.Y.type_as(monoms_patches), monoms_patches)
        
        sh_patches =  torch.einsum('ij,bvpj->bvpi', self.Y.type_as(monoms_patches), monoms_patches)
        # print(sh_patches.shape)
        #return sh_patches
        shells_rad = torch.arange(self.num_shells).type_as(monoms_patches) / (self.num_shells-1)

        shells_rad = torch.reshape(shells_rad, (1, 1, 1, -1))
        shells = patches_dist - shells_rad
        shells = torch.exp(-self.gaussian_scale*(shells * shells))
        shells_sum = torch.sum(shells, dim=-1, keepdims=True)
        shells = (shells / torch.maximum(shells_sum, torch.tensor(0.000001).type_as(shells)))

        shells = shells.unsqueeze(-2)
        if self.bound:
            shells = torch.where(patches_dist.unsqueeze(-1) <= torch.tensor(1.).type_as(shells), shells, torch.tensor(0.).type_as(shells))

        sh_patches = sh_patches.unsqueeze(-1)
        sh_patches = shells * sh_patches


        # L2 norm
        l2_norm = torch.sum((sh_patches * sh_patches), dim=2, keepdims=True)
        l2_norm = torch.split(l2_norm, split_size_or_sections=self.split_size, dim=-2)
        Y = []
        for l in range(len(l2_norm)):
            ml = torch.sum(l2_norm[l], dim=-2, keepdims=True)
            ml = torch.sqrt(ml + 1e-7)
            Y.append(ml)
        l2_norm = torch.cat(Y, dim=-2)
        l2_norm = torch.mean(l2_norm, dim=1, keepdims=True)
        l2_norm = torch.maximum(l2_norm, torch.tensor(1e-8).type_as(l2_norm))
        # print(l2_norm.shape)
        l2_norm = l2_norm[..., self.sh_idx, :]
        sh_patches = (sh_patches / (l2_norm + 1e-6))

        return sh_patches

class ShGaussianKernelConv_grid(torch.nn.Module):
    def __init__(self, l_max, l_max_out=None, transpose=False, num_source_points=None):
        super(ShGaussianKernelConv_grid, self).__init__()
        self.l_max = l_max
        self.split_size = []
        for l in range(l_max + 1):
            self.split_size.append(2 * l + 1)
        # self.output_type = output_type
        self.l_max_out = l_max_out
        # self.transpose = transpose
        self.num_source_points = num_source_points
        self.Q = torch_clebsch_gordan_decomposition(l_max=max(l_max_out, l_max),
                                                 sparse=False,
                                                 output_type='dict',
                                                 l_max_out=l_max_out)

    def forward(self, x):
        assert (isinstance(x, dict))
        
        signal = []
        features_type = []
        channels_split_size = []
        for l in x:
            if l.isnumeric():
                features_type.append(int(l))
                channels_split_size.append(x[l].shape[-2] * x[l].shape[-1])
                signal.append(torch.reshape(x[l], (x[l].shape[0], x[l].shape[1], -1)))


        signal = torch.cat(signal, dim=-1)
        batch_size = signal.shape[0]
        patch_size = x["kernels"].shape[2]
        num_shells = x["kernels"].shape[-1]

        # Changed and removed transpose here
        if "patches idx" in x:
            # print(signal.shape, "signal")
            B, N, _ = signal.shape
            H = int(np.cbrt(N))
            # print(x["patches idx"].shape, "idx")
            B2, N_s, K, _ = x["patches idx"].shape

            H_s = int(np.cbrt(N_s))
            
            signal = signal.reshape(B, H, H, H, -1).permute(0, 4, 1, 2, 3)
            
            id_sample = x["patches idx"]
            id_sample = id_sample.permute((0, -1, 1, 2))
            id_sample = id_sample.reshape(B, -1, H_s, H_s, H_s*K)
            id_sample = id_sample.permute((0, 2, 3, 4, 1))
            signal = torch.nn.functional.grid_sample(signal, id_sample, align_corners = True).reshape(B, -1, H_s, H_s, H_s, K).permute(0, 2, 3, 4, 5, 1)
            # print("patches mask found") 

            if "patches mask" in x:
                # print("patches mask found") 
                mask_patches = x["patches mask"]
                signal[mask_patches, :] = 0 

            signal = signal.reshape(B, H_s*H_s*H_s, K, -1)
            
            
        num_points_target = signal.shape[1]
        kernels = torch.reshape(x["kernels"], (batch_size, num_points_target, patch_size, -1))
        
        y = torch.einsum('bvpy,bvpc->bvyc', kernels, signal)



        # split y
        # print(channels_split_size, y.shape)
        y_ = torch.split(y, split_size_or_sections=channels_split_size, dim=-1)
        y = {str(j): [] for j in range(self.l_max_out + 1)}
        y_cg = []
        for i in range(len(channels_split_size)):
            l = features_type[i]
            yi = torch.reshape(y_[i], (batch_size, num_points_target, -1, num_shells, 2 * l + 1, x[str(l)].shape[-1]))
            yi = yi.permute(0, 1, 2, 4, 3, 5)
            yi = torch.reshape(yi, (batch_size, num_points_target, -1, 2 * l + 1, num_shells*x[str(l)].shape[-1]))
            yi = torch.split(yi, split_size_or_sections=self.split_size, dim=2)
            for j in range(len(self.split_size)):

                yij = yi[j]
                if l == 0:
                    y[str(j)].append(yij[:, :, :, 0, :])
                elif j == 0:
                    y[str(l)].append(yij[:, :, 0, :, :])
                else:
                    y_cg.append(yij)

        y_cg = self.Q.decompose(y_cg)


        for J in y_cg:
            if J not in y:
                y[J] = []
            y[J].append(y_cg[J])
        for J in y:
            y[J] = torch.cat(y[J], dim=-1)
        return y



if __name__=="__main__":

    gi = GroupPoints_density(0.4, 32)
    in_dict = {}
    x = torch.randn(2, 32 *32 *32, 3).cuda()
    y = torch.randn(2, 16 *16 *16, 3).cuda()
    x_d = torch.randn(2, 32 *32 *32, 1).cuda()
    y_d = torch.randn(2, 16 *16 *16, 1).cuda()
    in_dict["source points"] = x
    in_dict["target points"] = y
    in_dict["source density"] = x_d
    in_dict["target density"] = y_d
    x2 = y
    out = gi(in_dict)
    for key in out:
        print(key, " ", out[key].shape)
    B = x.shape[0]
    H, W, D= 32, 32, 32
    K = out["patches source"].shape[-2]
    # B, H, W, D, K, _ = out["patches source"].shape
    k = SphericalHarmonicsGaussianKernels_density(l_max = 3, gaussian_scale = 0.1, num_shells = 3, bound = True).cuda()
    print(out["patches source"].shape)
    out_2 = k({"patches": out["patches source"].reshape(B, -1, K, 3), "patches dist": out["patches dist source"].reshape(B, -1, K), "patches_density": out["patches source density"].reshape(B, -1, K, 1)}) # B, H*W*D, K, 16, 3

    conv_layer = ShGaussianKernelConv_grid(l_max=3, l_max_out=3).cuda()
    y = {}
    x2 = x2.reshape(B, -1, 3)
    y["source points"] = x.reshape(B, -1, 3)
    y["target points"] = x2
    y["patches idx"] = out["patches idx source"].reshape(B, -1, K, 3)
    y["patches dist source"] = out["patches dist source"].reshape(B, -1, K)
    y["kernels"] = out_2
    # w = gauss_normalization(y["patches dist source"], 1./3.)

    

    if '1' in y:
        y['1'] = torch.cat([y['1'], x2], dim=-1)
    else:
        y['1'] = x2.unsqueeze(-1)

    y = conv_layer(y)

    # for key in y:
    #     print(y[key], " ", key, " ", y[key].shape)
    
