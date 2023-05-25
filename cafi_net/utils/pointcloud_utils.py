import torch
from utils.group_points import gather_idx
import numpy as np
import h5py
import open3d as o3d
import seaborn as sns
from spherical_harmonics.spherical_cnn import zernike_monoms
from spherical_harmonics.spherical_cnn import torch_fibonnacci_sphere_sampling
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)
    
def compute_centroids(points, capsules):
    return torch.einsum('bij,bic->bcj', points, capsules)
    
def sq_distance_mat(pcd_1, pcd_2):

    r0 = pcd_1 * pcd_1
    r0 = torch.sum(r0, dim=2, keepdims=True)
    r1 = (pcd_2 * pcd_2)
    r1 = torch.sum(r1, dim=2, keepdims=True)
    r1 = r1.permute(0, 2, 1)
    sq_distance_mat = r0 - 2. * (pcd_1 @ pcd_2.permute(0, 2, 1)) + r1

    return sq_distance_mat.squeeze(-1)
    
def convert_yzx_to_xyz_basis(basis):

    # basis - N, 3, 3

    rot_y = torch.tensor([[np.cos(np.pi / 2), 0, np.sin(np.pi / 2)]
              ,[0,              1,                      0], 
              [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]]).type_as(basis)


    rot_z = torch.tensor([
                    [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
                    [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                    [0, 0, 1]
                    ]).type_as(basis)

    transform = (rot_y @ rot_z).unsqueeze(0).type_as(basis)
    transform = transform.repeat(basis.shape[0], 1, 1)
    
    if len(basis.shape) == 4:
        transform = transform.unsqueeze(1)

    return transform @ basis

def create_color_samples(N):
    '''
    Creates N distinct colors
    N x 3 output
    '''

    palette = sns.color_palette(None, N)
    palette = np.array(palette)

    return palette

def convert_tensor_2_numpy(x):
    '''
    Convert pytorch tensor to numpy
    '''
    
    out = x.squeeze(0).detach().cpu().numpy()
    
    return out 

def save_density(x, filename = "./pointcloud.ply"):
    density_int = x.cpu().numpy()
    shape_ = density_int.shape 
    density_int = density_int.reshape(-1,1)
    mask_density_int = np.ones_like(density_int) * -1
        
    model = KMeans(init="k-means++",n_clusters=2)
    model.fit(density_int)
    label = model.predict(density_int)
    clusters = np.unique(label)
    
    if np.mean(density_int[np.where(label == 1)[0]]) > np.mean(density_int[np.where(label == 0)[0]]):
        fg_idx = np.where(label == 1)[0]
        bg_idx = np.where(label == 0)[0]


    else:
        fg_idx = np.where(label == 0)[0]
        bg_idx = np.where(label == 1)[0]

    mask_density_int[fg_idx] = 1.
    mask_density_int[bg_idx] = 0.
        
    mask_density_tensor = torch.from_numpy(mask_density_int)
    mask_density_int = mask_density_int.reshape(shape_)
    sampling_grid = get_xyz_grid(x)
    sampling_grid_np = sampling_grid.detach().cpu().numpy()
    pts = sampling_grid_np[mask_density_int == 1., :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    label_map = np.ones((pts.shape[0], 3)) * 0.5
    pcd.colors = o3d.utility.Vector3dVector(label_map)
    o3d.io.write_point_cloud(filename, pcd)
   


def save_pointcloud(x, filename = "./pointcloud.ply"):
    '''
    Save point cloud to the destination given in the filename
    x can be list of inputs (Nx3) capsules or numpy array of N x 3
    '''

    label_map = []
    if isinstance(x, list):
        
        pointcloud = []
        labels = create_color_samples(len(x))
        for i in range(len(x)):
            pts = x[i]
            # print(pts.shape, "vis")
            pts = convert_tensor_2_numpy(pts)

            pointcloud.append(pts)
            label_map.append(np.repeat(labels[i:(i + 1)], pts.shape[0], axis = 0))
        
        # x = np.concatenate(x, axis = 0)
        pointcloud = np.concatenate(pointcloud, axis = 0)
        x = pointcloud.copy()
        label_map = np.concatenate(label_map, axis = 0)
    else:
        x = convert_tensor_2_numpy(x)
        label_map = np.ones((len(x), 3)) * 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    pcd.colors = o3d.utility.Vector3dVector(label_map)

    o3d.io.write_point_cloud(filename, pcd)

def diameter(x, axis=-2, keepdims=True):
    return torch.max(x, dim=axis, keepdims=keepdims).values - torch.min(x, dim=axis, keepdims=keepdims).values

def kdtree_indexing(x, depth=None, return_idx = False):
    num_points = x.shape[1]
    #assert isPowerOfTwo(num_points)
    if depth is None:
        depth = int(np.log(num_points) / np.log(2.) + 0.1)
    y = x
    batch_idx = torch.arange(x.shape[0]).to(torch.int32).to(x.device)
    batch_idx = torch.reshape(batch_idx, (-1, 1))
    batch_idx = batch_idx.repeat(1, x.shape[1])

    points_idx = torch.arange(num_points).type_as(x).to(torch.int64)
    points_idx = torch.reshape(points_idx, (1, -1, 1))
    points_idx = points_idx.repeat(x.shape[0], 1, 1)



    for i in range(depth):
        y_shape = list(y.shape)
        diam = diameter(y)
        split_idx = torch.argmax(diam, dim=-1).to(torch.long).to(x.device)
        split_idx = split_idx.repeat(1, y.shape[1])
        idx = torch.arange(y.shape[0]).to(torch.long).to(x.device)
        idx = idx.unsqueeze(-1)
        idx = idx.repeat(1, y.shape[1])
        branch_idx = torch.arange(y.shape[1]).to(torch.long).to(x.device)
        branch_idx = branch_idx.unsqueeze(0)
        branch_idx = branch_idx.repeat(y.shape[0], 1)
        split_idx = torch.stack([idx, branch_idx, split_idx], dim=-1)
        # print(split_idx, split_idx.shape)
        # Gather implementation required
        # m = tf.gather_nd(y, split_idx)
        # print(y.shape)
        m = gather_idx(y, split_idx)
        # print(m.shape)
        sort_idx = torch.argsort(m, dim=-1)
        sort_idx = torch.stack([idx, sort_idx], dim=-1)
        # Gather required
        points_idx = gather_idx(points_idx, sort_idx)
        points_idx = torch.reshape(points_idx, (-1, int(y.shape[1] // 2), 1))
        # Gather
        y = gather_idx(y, sort_idx)
        y = torch.reshape(y, (-1, int(y.shape[1] // 2), 3))


    
    y = torch.reshape(y, x.shape)
    if not return_idx:
        return y

    points_idx = torch.reshape(points_idx, (x.shape[0], x.shape[1]))
    points_idx_inv = torch.argsort(points_idx, dim=-1)
    points_idx = torch.stack([batch_idx, points_idx], dim=-1)
    points_idx_inv = torch.stack([batch_idx, points_idx_inv], dim=-1)

    return y, points_idx, points_idx_inv

def kdtree_indexing_sdf(x, sdf, depth=None, return_idx = False):
    num_points = x.shape[1]
    #assert isPowerOfTwo(num_points)
    if depth is None:
        depth = int(np.log(num_points) / np.log(2.) + 0.1)
    y = x
    batch_idx = torch.arange(x.shape[0]).to(torch.int32).to(x.device)
    batch_idx = torch.reshape(batch_idx, (-1, 1))
    batch_idx = batch_idx.repeat(1, x.shape[1])

    points_idx = torch.arange(num_points).type_as(x).to(torch.int64)
    points_idx = torch.reshape(points_idx, (1, -1, 1))
    points_idx = points_idx.repeat(x.shape[0], 1, 1)



    for i in range(depth):
        y_shape = list(y.shape)
        diam = diameter(y)
        split_idx = torch.argmax(diam, dim=-1).to(torch.long).to(x.device)
        split_idx = split_idx.repeat(1, y.shape[1])
        idx = torch.arange(y.shape[0]).to(torch.long).to(x.device)
        idx = idx.unsqueeze(-1)
        idx = idx.repeat(1, y.shape[1])
        branch_idx = torch.arange(y.shape[1]).to(torch.long).to(x.device)
        branch_idx = branch_idx.unsqueeze(0)
        branch_idx = branch_idx.repeat(y.shape[0], 1)
        split_idx = torch.stack([idx, branch_idx, split_idx], dim=-1)
        # print(split_idx, split_idx.shape)
        # Gather implementation required
        # m = tf.gather_nd(y, split_idx)
        # print(y.shape)
        m = gather_idx(y, split_idx)
        # print(m.shape)
        sort_idx = torch.argsort(m, dim=-1)
        sort_idx = torch.stack([idx, sort_idx], dim=-1)
        # Gather required
        points_idx = gather_idx(points_idx, sort_idx)
        points_idx = torch.reshape(points_idx, (-1, int(y.shape[1] // 2), 1))
        # Gather
        y = gather_idx(y, sort_idx)
        y = torch.reshape(y, (-1, int(y.shape[1] // 2), 3))


    
    y = torch.reshape(y, x.shape)
    if not return_idx:
        return y

    points_idx = torch.reshape(points_idx, (x.shape[0], x.shape[1]))
    points_idx_inv = torch.argsort(points_idx, dim=-1)
    points_idx = torch.stack([batch_idx, points_idx], dim=-1)
    points_idx_inv = torch.stack([batch_idx, points_idx_inv], dim=-1)

    return y, points_idx, points_idx_inv





def get_xyz_grid(grid, rotation = None):
    """
    Returns an xyz grid in the normalized coordinates to perform density convolution
    
    grid - B, C, H, W, D
    
    out - B, C, H, W, 3
    """

    if len(grid.shape) == 4:
        out = grid.unsqueeze(1)
    else:
        out = grid

    if rotation is not None:
        theta = rotation
    else: 
        theta = torch.eye(3).unsqueeze(0).repeat(out.shape[0], 1, 1).type_as(grid)
    t = torch.tensor([0, 0, 0]).unsqueeze(0).unsqueeze(2).repeat(theta.shape[0], 1, 1).type_as(grid)
    theta = torch.cat([theta, t], dim = -1)
    out_grid = torch.nn.functional.affine_grid(theta, out.shape, align_corners = True)

    return out_grid

def get_equivariant_density(density_field, normalize = True, scale_grid = True):
    """
    Get equivariant density grid
    density_field - B, H, W, D
    """

    out = density_field
    if len(density_field.shape) == 4:
        out = density_field.unsqueeze(1)

    B, _1, H, W, D = out.shape

    grid = get_xyz_grid(density_field) # B, H, W, D, 3
    grid_pts = grid.reshape(B, H*W*D, 3)

    out_density = out.reshape(B, H*W*D, 1)
    if normalize:
        density_max = torch.max(out_density, axis = 1, keepdims = True)
        out_density = out_density / (density_max.values + 1e-8)
    
    # eq_grid_pts = zernike_monoms(grid_pts, 3)[1][..., 1:] # B, N, 3, 1
    # eq_grid_pts = out_density * eq_grid_pts[..., 0] # B, N, 3
    eq_grid_pts = grid_pts.unsqueeze(-1) #zernike_monoms(grid_pts, 3)[1][..., 1:] # B, N, 3, 1
    if scale_grid:
        eq_grid_pts = out_density * eq_grid_pts[..., 0] # B, N, 3
    else:
        eq_grid_pts = eq_grid_pts[..., 0]
        
    eq_grid_pts = eq_grid_pts.reshape(B, H, W, D, 3)

    return eq_grid_pts


class GroupPoints_grid(torch.nn.Module):
    def __init__(self, radius, patch_size_source, radius_target=None, patch_size_target=None,
                 spacing_source=0, spacing_target=0):
        super(GroupPoints_grid, self).__init__()

        """
        Group points and different scales for pooling
        """
        self.radius = radius
        self.radius_target = radius_target
        self.patch_size_source = patch_size_source
        self.patch_size_target = patch_size_target
        self.spacing_source = spacing_source
        self.spacing_target = spacing_target
        self.sphere_sampling = torch_fibonnacci_sphere_sampling(patch_size_source) # 14, 3
        # self.sphere_sampling = torch.cat([self.sphere_sampling, self.sphere_sampling*2, self.sphere_sampling*3], 0)

    def forward(self, x):
        """
        source, target - B, H, W, D, 3

        :param x: [source, target]
        :return: [patches_idx_source, num_incident_points_target]
        Returns:
            source patches - B, N, K, 3
            patches idx source - B, N, K, 2
            patches size source - B, N
            patches radius source - B, 1, 1
            patches dist source - B, N, K
        """
        assert isinstance(x, dict)
        source = x["source points"]
        target = x["target points"]

        B, N_s, _ = source.shape
        B, N_t, _ = target.shape

        H_s = int(np.cbrt(N_s))
        H_t = int(np.cbrt(N_t))
        source = source.reshape(B, H_s, H_s, H_s, 3)
        target = target.reshape(B, H_t, H_t, H_t, 3)
        self.sphere_sampling = self.sphere_sampling.type_as(source)

        source_mask = None
        if "source mask" in x:
            source_mask = x["source mask"]

        target_mask = None
        if "target mask" in x:
            target_mask = x["target mask"]

        num_points_source = source.shape[1]

        # assert (num_points_source >= self.patch_size_source)
        if self.patch_size_target is not None:
            num_points_target = target.shape[1]
            # assert (num_points_target >= self.patch_size_source)

        # Compute spheres of radius around each point
        grid_target = get_xyz_grid(target[..., 0].unsqueeze(1))
        # grid_target = get_xyz_grid(target) # B, H, W, D, 3
        grid_target = grid_target.unsqueeze(-2).type_as(source) # B, H, W, D, 1, 3
        B, H, W, D, _, _ = grid_target.shape
        sphere_samples = self.sphere_sampling.reshape(1, 1, 1, 1, -1, 3).type_as(source)
        sampling_grid = grid_target + sphere_samples / (grid_target.shape[1] + 1e-8) * self.radius # B, H, W, D, patch_size, 3
        sampling_grid = sampling_grid.permute((0, -1, 1, 2, 3, 4))
        sampling_grid = sampling_grid.reshape(B, 3, H, W, -1)
        sampling_grid = sampling_grid.permute((0, 2, 3, 4, 1))
        # sampling_grid = sampling_grid.reshape(B, H, W, -1, 3)

        patches = torch.nn.functional.grid_sample(target.transpose(1, -1), sampling_grid, align_corners = True)
        patches = patches.reshape(B, -1, H, W, D, self.patch_size_source)
        patches = patches.permute(0, 2, 3, 4, 5, 1)
        # patches = patches.reshape(B, H, W, D, self.patch_size_source, 3)
        sampling_grid = sampling_grid.permute((0, 4, 1, 2, 3))
        sampling_grid = sampling_grid.reshape(B, 3, H, W, D, -1)
        sampling_grid = sampling_grid.permute((0, 2, 3, 4, 5, 1))
        y = {}
        y["patches source"] = patches
        y["patches idx source"] = sampling_grid
        patches_dist = torch.sum(torch.square(target.unsqueeze(-2) - patches), axis = -1) # B, H, W, D, patch, 3
        patches_dist = torch.sqrt(torch.maximum(patches_dist, torch.tensor(0.000000001).type_as(patches_dist)))
        y["patches dist source"] = patches_dist

        return y



def get_gradient_density(x):
    
    # x - B, H, W, D
    
    B, H, W, D = x.shape
    
    data = x.clone()
    data = data.cpu().detach().numpy()
    
    x_grad = torch.from_numpy(np.gradient(data,axis=3,edge_order=2)).to(x.device)
    y_grad = torch.from_numpy(np.gradient(data,axis=2,edge_order=2)).to(x.device)
    z_grad = torch.from_numpy(np.gradient(data,axis=1,edge_order=2)).to(x.device)
    
   
    gradient = torch.stack([y_grad, z_grad, x_grad], 1)

    return gradient # B, 3, H, W, D

class GroupPoints_density(torch.nn.Module):
    def __init__(self, radius, patch_size_source, radius_target=None, patch_size_target=None,
                 spacing_source=0, spacing_target=0):
        super(GroupPoints_density, self).__init__()

        """
        Group points and different scales for pooling
        """
        #import pdb
        #pdb.set_trace()
        self.radius = radius
        self.radius_target = radius_target
        self.patch_size_source = patch_size_source 
        self.patch_size_target = patch_size_target #* 3
        self.spacing_source = spacing_source
        self.spacing_target = spacing_target
        self.sphere_sampling = torch_fibonnacci_sphere_sampling(patch_size_source) # 14, 3
        
        # print(self.sphere_sampling.shape)

    def forward(self, x):
        """
        source, target - B, H, W, D, 3

        :param x: [source, target]
        :return: [patches_idx_source, num_incident_points_target]
        Returns:
            source patches - B, N, K, 3
            patches idx source - B, N, K, 2
            patches size source - B, N
            patches radius source - B, 1, 1
            patches dist source - B, N, K
        """
        
        assert isinstance(x, dict)
        source = x["source points"]
        target = x["target points"]
        source_density = x["source density"]
        target_density = x["target density"]

        B, N_s, _ = source.shape
        B, N_t, _ = target.shape

        H_s = int(np.cbrt(N_s))
        H_t = int(np.cbrt(N_t))
        source = source.reshape(B, H_s, H_s, H_s, 3)
        target = target.reshape(B, H_t, H_t, H_t, 3)
        source_density = source_density.reshape(B, H_s, H_s, H_s, 1)
        target_density = target_density.reshape(B, H_t, H_t, H_t, 1)

        self.sphere_sampling = self.sphere_sampling.type_as(source)
        rad = self.radius * torch.ones((B, 1, 1, 1, 1)).type_as(source)
        
    
        # Compute spheres of radius around each point
        grid_target = get_xyz_grid(target[..., 0].unsqueeze(1))
        # grid_target = get_xyz_grid(target) # B, H, W, D, 3
        grid_target = grid_target.unsqueeze(-2).type_as(source) # B, H, W, D, 1, 3
        B, H, W, D, _, _ = grid_target.shape
        sphere_samples = self.sphere_sampling.reshape(1, 1, 1, 1, -1, 3).type_as(source)# / (rad.squeeze() + 1e-8)
        sampling_grid = grid_target + sphere_samples / (grid_target.shape[1] + 1e-8) / (rad.unsqueeze(-2) + 1e-6) # B, H, W, D, patch_size, 3
        sampling_grid = sampling_grid.permute((0, -1, 1, 2, 3, 4))
        sampling_grid = sampling_grid.reshape(B, 3, H, W, -1)
        sampling_grid = sampling_grid.permute((0, 2, 3, 4, 1))
        # sampling_grid = sampling_grid.reshape(B, H, W, -1, 3)

        source = source / (rad + 1e-6)
        target = target / (rad + 1e-6)

        patches = torch.nn.functional.grid_sample(source.transpose(1, -1), sampling_grid, align_corners = True)
        patches_density = torch.nn.functional.grid_sample(source_density.transpose(1, -1), sampling_grid, align_corners = True)
        # print(patches_density.shape)
        patches_density = patches_density.reshape(B, 1, H, W, D, self.patch_size_source)
        patches_density = patches_density.permute(0, 2, 3, 4, 5, 1) # B, H, W, D, K, 1
        patches = patches.reshape(B, -1, H, W, D, self.patch_size_source) 
        patches = patches.permute(0, 2, 3, 4, 5, 1) # B, H, W, D, K, 3
        _,_,_,_,d,c = patches.shape
        patches_flat = patches.reshape(1,-1,d,c)
        patches = patches - target.unsqueeze(-2) # B, H, W, D, K, 3
        # patches_density = torch.abs(patches_density - target_density.unsqueeze(-2)) # B, H, W, D, K, 1
        patches_density = torch.sqrt( 1e-8 + (patches_density * target_density.unsqueeze(-2))) # B, H, W, D, K, 1
        # patches_density = patches_density # B, H, W, D, K, 1
        
        # patches = patches.reshape(B, H, W, D, self.patch_size_source, 3)
        sampling_grid = sampling_grid.permute((0, 4, 1, 2, 3))
        sampling_grid = sampling_grid.reshape(B, 3, H, W, D, -1)
        sampling_grid = sampling_grid.permute((0, 2, 3, 4, 5, 1))
        patches_dist = torch.sum(torch.square(patches), axis = -1) # B, H, W, D, patch, 3
        patches_dist = torch.sqrt(torch.maximum(patches_dist, torch.tensor(0.000000001).type_as(patches_dist)))

        mask = torch.lt(rad.type_as(patches_dist) ** 2, (patches_dist * rad)**2)
        # print(mask.shape)

        # sampling_grid[mask, :] = -10
        # Reject where mask is 1
        patches[mask, :] = 0
        patches_density[mask, :] = 0
        
        y = {}
        y["patches source"] = patches
        y["patches source density"] = patches_density # B, H, W, D, K, 1
        y["patches idx source"] = sampling_grid
        # patches_dist = patches_dist #/ (rad + 1e-6)
        y["patches dist source"] = patches_dist
        y["patches mask"] = mask
        y["patches size"] = torch.sum( 1 - 1*mask, -1)

        return y


def rotate_density(rotation, density_field, affine = True):
    """
    rotation - B, 3, 3
    density_field - B, H, W, D or B, C, H, W, D
    """
    if len(density_field.shape) == 4:
        out = density_field.unsqueeze(1)
    else:
        out = density_field

    rotation = rotation.type_as(density_field)
    t = torch.tensor([0, 0, 0]).unsqueeze(0).unsqueeze(2).repeat(rotation.shape[0], 1, 1).type_as(density_field)
    theta = torch.cat([rotation, t], dim = -1)
    if affine == True:
        rot_grid = torch.nn.functional.affine_grid(theta, out.shape, align_corners = True)
    else:
        x = torch.linspace(-1, 1, density_field.shape[2]).type_as(density_field)
        grid = torch.stack(torch.meshgrid(x, x, x), axis = -1).unsqueeze(0).repeat(out.shape[0], 1, 1, 1, 1) 
        # print(grid.shape, rotation.shape)

        rot_grid = torch.einsum("bij, bhwdj-> bhwdi", rotation, grid)
    #print(rot_grid)
    rotated_grid = torch.nn.functional.grid_sample(out, rot_grid, align_corners = True, mode="nearest")#, padding_mode = "border")

    if len(density_field.shape) == 4:
        rotated_grid = rotated_grid.squeeze(1)
        
    return rotated_grid

def patches_radius(radius, sq_norm):
    batch_size = sq_norm.shape[0]
    rad = radius
    if isinstance(radius, float):
        rad = radius * torch.ones((batch_size, 1, 1))
    if isinstance(radius, str):
        rad = torch.sqrt(torch.maximum(torch.max(sq_norm, dim=2, keepdims=False), torch.tensor(0.0000001).type_as(sq_norm)))
        if radius == "avg":
            rad = torch.mean(rad, dim=-1, keepdims=False)
        elif radius == 'min':
            rad = torch.min(rad, dim=-1, keepdims=False)
        elif radius.isnumeric():
            rad = torch.sort(rad, dim=-1)
            i = int((float(int(radius)) / 100.) * sq_norm.shape[1])
            i = max(i, 1)
            rad = torch.mean(rad[:, :i], dim=-1, keepdims=False)
        rad = torch.reshape(rad, (batch_size, 1, 1))
    return rad


def gather_idx(x, idx):


    """
    x - B, N, 3
    idx - B, N, K, 2/3

    out - B, N, K, 3
    """
    num_idx = idx.shape[-1]
    
    if idx.shape[-1] == 3:
        if len(x.shape) == 3:
            out = x[idx[..., 0], idx[..., 1], idx[..., 2]]
            out[(idx[..., 2] < 0) * (idx[..., 1] < 0)] = 0
            return out

    if len(x.shape) == 2:
        out = x[idx[..., 0], idx[..., 1]]
        out[idx[..., 1] < 0] = 0
    else:
        out = x[idx[..., 0], idx[..., 1], :]
        out[idx[..., 1] < 0, :] = 0

    # print(idx[..., 1].shape, out.shape)

    return out
    
    
def gather_idx_density(x, idx):


    """
    x - B, N, 3
    idx - B, N, K, 2/3

    out - B, N, K, 3
    """
    num_idx = idx.shape[-1]
    
    if idx.shape[-1] == 3:
        if len(x.shape) == 3:
            out = x[idx[..., 0], idx[..., 1], idx[..., 2]]
            out[(idx[..., 2] < 0) * (idx[..., 1] < 0)] = 0
            return out

    if len(x.shape) == 2:
        out = x[idx[..., 0], idx[..., 1]]
        out[idx[..., 1] < 0] = 0
    else:
        out = x[idx[..., 0], idx[..., 1], :]
        out[idx[..., 1] < 0, :] = 0

    # print(idx[..., 1].shape, out.shape)

    return out

def compute_patches(source, target, sq_distance_mat, sq_distance_mat_sel, num_samples, spacing, radius, source_mask=None,source_density=None):
    
    batch_size = source.shape[0]
    num_points_source = source.shape[1]
    num_points_target = target.shape[1]
    assert (num_samples * (spacing + 1) <= num_points_source)
    
    sq_patches_dist, patches_idx = torch.topk(-sq_distance_mat, k=num_samples * (spacing + 1))
    
    #B,N,_  = sq_distance_mat.shape
    
    #batch_idx         =  torch.arange(0,B).repeat(patches_idx.shape[-1],1).T.unsqueeze(1).repeat(1,N,1).to(torch.int64)
    #point_idx         = torch.arange(0,N).reshape(N,1).repeat(1,patches_idx.shape[-1]).unsqueeze(0).repeat(B,1,1).to(torch.int64)
    #sq_patches_dist   = sq_distance_mat[batch_idx,point_idx,patches_idx]
    
    sq_patches_dist = -sq_patches_dist
    if spacing > 0:
        sq_patches_dist = sq_patches_dist[:, :, 0::(spacing + 1), ...]
        patches_idx = patches_idx[:, :, 0::(spacing + 1), ...]

    rad = patches_radius(radius, sq_patches_dist).type_as(sq_distance_mat)
    patches_size = patches_idx.shape[-1]

    # mask = sq_patches_dist < radius ** 2
    mask = torch.greater_equal(rad.type_as(sq_distance_mat) ** 2, sq_patches_dist)
    patches_idx = (torch.where(mask, patches_idx, torch.tensor(-1).type_as(patches_idx))).to(torch.int64)
    if source_mask is not None:
        source_mask = source_mask < 1
        source_mask = source_mask.unsqueeze(-1).repeat(1, 1, patches_idx.shape[-1])
        patches_idx = torch.where(source_mask, patches_idx, torch.tensor(-1).type_as(patches_idx))

    batch_idx = torch.arange(batch_size).type_as(patches_idx)
    batch_idx = torch.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = batch_idx.repeat(1, num_points_target, num_samples)
    patches_idx = torch.stack([batch_idx, patches_idx], dim = -1).to(torch.long)

    source = (source / (rad + 1e-6))
    target = (target / (rad + 1e-6))
    
    # patches = source[batch_idx.to(torch.long), patches_idx.to(torch.long)]
    patches = gather_idx(source, patches_idx)
    
    b,n,c,_ = patches.shape
    #density = torch.ones(b,n,c,1)
    density = gather_idx_density(source_density, patches_idx)
    # patches = source[patches_idx[..., 0], patches_idx[..., 1], :]
    # print(patches.shape, "patch")
    patches = patches - target.unsqueeze(-2)
    
    
    


    

    if source_mask is not None:
        mask = source_mask
    else:
        mask = torch.ones((batch_size, num_points_source)).type_as(patches)
    
    patch_size = gather_idx(mask, patches_idx.to(torch.long))
    # patch_size = mask[patches_idx[..., 0], patches_idx[..., 1]]
    patches_size = torch.sum(patch_size, dim=-1, keepdims=False)
    patches_dist = torch.sqrt(torch.maximum(sq_patches_dist, torch.tensor(0.000000001).type_as(sq_patches_dist)))
    patches_dist = patches_dist / (rad + 1e-6)
    
    mask_cnt     =  torch.count_nonzero(patches_idx[..., 1] <0,dim=-1).unsqueeze(-1)
    weight_density= torch.sum(density,dim=-2) / mask_cnt
    
    return {"patches": patches, "patches idx": patches_idx, "patches size": patches_size, "patches radius": rad,
            "patches dist": patches_dist, "source_density":density, "weight_density":weight_density}



def create_mask(density):
    
    shape_ = density.shape
    
    B,H,W,D = density.shape
    mask_density_list = []
    for i in range(shape_[0]):
        
        density_int = density[i].cpu().numpy().reshape(-1,1)
        mask_density_int = np.ones_like(density_int) * -1
        
        model = KMeans(init="k-means++",n_clusters=2)
        model.fit(density_int)
        label = model.predict(density_int)
        clusters = np.unique(label)
        
        if np.mean(density_int[np.where(label == 1)[0]]) > np.mean(density_int[np.where(label == 0)[0]]):
            fg_idx = np.where(label == 1)[0]
            bg_idx = np.where(label == 0)[0]


        else:
            fg_idx = np.where(label == 0)[0]
            bg_idx = np.where(label == 1)[0]

        mask_density_int[fg_idx] = 1.
        mask_density_int[bg_idx] = 0.
            
        mask_density_tensor = torch.from_numpy(mask_density_int)
        mask_density_list.append(mask_density_tensor)
    mask_density = torch.stack(mask_density_list).reshape(shape_)
    return mask_density 


class GroupPoints_euclidean_density(torch.nn.Module):
    def __init__(self, radius, patch_size_source, radius_target=None, patch_size_target=None,
                 spacing_source=0, spacing_target=0):
        super(GroupPoints_euclidean_density, self).__init__()

        """
        Group points and different scales for pooling
        """
        
        self.radius = radius
        self.radius_target = radius_target
        self.patch_size_source = patch_size_source 
        self.patch_size_target = patch_size_target #* 3
        self.spacing_source = spacing_source
        self.spacing_target = spacing_target
        self.sphere_sampling = torch_fibonnacci_sphere_sampling(patch_size_source) # 14, 3
        #self.sphere_sampling = torch.cat([self.sphere_sampling, self.sphere_sampling*2, self.sphere_sampling*3], 0)
        

    def forward(self, x):
        """
        source, target - B, H, W, D, 3

        :param x: [source, target]
        :return: [patches_idx_source, num_incident_points_target]
        Returns:
            source patches - B, N, K, 3
            patches idx source - B, N, K, 2
            patches size source - B, N
            patches radius source - B, 1, 1
            patches dist source - B, N, K
        """
        
        assert isinstance(x, dict)
        source = x["source points"]
        target = x["target points"]
        source_density = x["source density"]
        target_density = x["target density"]
        source_mask = None
        if "source mask" in x:
            source_mask = x["source mask"]
        #source =  (target.unsqueeze(-2) + self.sphere_sampling).reshape(1,-1,3).to(torch.float64)
        #source = self.sphere_sampling.unsqueeze(0).to(torch.float64)
        B, N_s, _ = source.shape
        B, N_t, _ = target.shape
        num_points_source = N_s
        
        # compute distance mat
        r0 = target * target
        r0 = torch.sum(r0, dim=2, keepdims=True)
        r1 = (source * source)
        r1 = torch.sum(r1, dim=2, keepdims=True)
        r1 = r1.permute(0, 2, 1)
        sq_distance_mat = r0 - 2. * (target @ source.permute(0, 2, 1)) + r1
        sq_distance_mat_sel = sq_distance_mat/((target_density @ source_density.permute(0,2,1)) + 1e-8)
        
        # Returns
        
        '''
        mask_density = torch.ones_like(source_density)
        mask_density[source_density == 0]= 1e10
        mask_density = mask_density.squeeze(-1)
        mask_denisty = mask_density.unsqueeze(-2)
        sq_distance_mat = sq_distance_mat * mask_density.unsqueeze(-2)'''
        
        #index = torch.where(source_density.squeeze(-1) == 0)
        #sq_distance_mat[index[0],:,index[1]] = 1e10
        
        patches = compute_patches(source, target, sq_distance_mat,sq_distance_mat_sel,
                                   min(self.patch_size_source, num_points_source),
                                   self.spacing_source, self.radius,
                                   source_mask=source_mask,source_density=source_density)
        
        
        H_t = int(np.cbrt(N_t))
        
        y = {}
        # concatinate points and density
        B,N,K,_ = patches["patches"].shape
        y["patches source"] = patches["patches"].reshape(B,H_t,H_t,H_t,K,-1) # B, N, K, 3
        #y["patches source density"] = patches["source_density"].reshape(B,H_t,H_t,H_t,K,-1)
        y["patches idx source"] = patches["patches idx"].reshape(B,H_t,H_t,H_t,K,-1)
        y["patches size source"] = patches["patches size"]
        y["patches radius source"] = patches["patches radius"]
        y["patches dist source"] = patches["patches dist"].reshape(B,H_t,H_t,H_t,K)
        y["weighted density"]    = patches["weight_density"]
        return y

if __name__=="__main__":


    # gi = GroupPoints_density(0.4, 32)
    # in_dict = {}
    # x = torch.randn(2, 32 *32 *32, 3).cuda()
    # y = torch.randn(2, 16 *16 *16, 3).cuda()
    # x_d = torch.randn(2, 32 *32 *32, 1).cuda()
    # y_d = torch.randn(2, 16 *16 *16, 1).cuda()
    # in_dict["source points"] = x
    # in_dict["target points"] = y
    # in_dict["source density"] = x_d
    # in_dict["target density"] = y_d

    # out = gi(in_dict)
    # for key in out:
    #     print(key, " ", out[key].shape)

    #x = torch.randn(2, 32, 32, 32).cuda()
    # out = get_gradient_density(x)
    #out = rotate_density(torch.eye(3).unsqueeze(0).repeat(2, 1, 1), x)
    #out_2 = rotate_density(torch.eye(3).unsqueeze(0).repeat(2, 1, 1), x, affine = False)
    # print(out - out_2)
    # print(torch.norm(out, dim = 1).shape)
    gi = GroupPoints_density(0.4, 32)
    in_dict = {}
    x = torch.randn(2, 32 *32 *32, 3)#.cuda()
    y = torch.randn(2, 16 *16 *16, 3)#.cuda()
    x_d = torch.randn(2, 32 *32 *32, 1)#.cuda()
    y_d = torch.randn(2, 16 *16 *16, 1)#.cuda()
    in_dict["source points"] = x
    in_dict["target points"] = y
    in_dict["source density"] = x_d
    in_dict["target density"] = y_d

    out = gi(in_dict)
    for key in out:
        print(key, " ", out[key].shape)
    # # x = (torch.ones((2, 1024, 3)) * torch.reshape(torch.arange(1024), (1, -1, 1))).cuda()
    # # x = torch.randn((2, 1024, 3)).cuda()
    # filename = "/home/rahul/research/data/sapien_processed/train_refrigerator.h5"
    # f = h5py.File(filename, "r")
    # x = torch.from_numpy(f["data"][:2]).cuda()
    # # x2 = torch.from_numpy(f["data"][2:4]).cuda()
    # y, kd, kd_2 = kdtree_indexing(x, return_idx = True)
    # print(x, x.shape, y, y.shape)

    # print(kd, kd.sha
