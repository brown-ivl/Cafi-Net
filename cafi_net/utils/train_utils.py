from scipy.spatial.transform import Rotation
import torch


def compute_centroids(points, capsules):
    return torch.einsum('bij,bic->bcj', points, capsules)
    
    
def normalize_caps(caps, eps = 1e-8):

    caps_sum = torch.sum(caps, dim = 1, keepdims = True)

    # Normalizing capsules
    normalized_caps = torch.divide(caps, caps_sum + eps)

    return normalized_caps
    
    
def random_rotate(x):

    """
    x - B, N, 3
    out - B, N, 3
    Randomly rotate point cloud
    """
    
    out = perform_rotation(torch.from_numpy(Rotation.random(x.shape[0]).as_matrix()).type_as(x), x)

    return out

def mean_center(x):
    """
    x - B, N, 3
    x_mean - B, N, 3
    Mean center point cloud
    """

    out = x - x.mean(-2, keepdims = True)
    return out

def perform_rotation(R, x):
    '''
    Perform rotation on point cloud
    R - B, 3, 3
    x - B, N, 3

    out - B, N, 3
    '''
    out = torch.einsum("bij,bpj->bpi", R.type_as(x), x)

    return out

def orthonormalize_basis(basis):
    """
    Returns orthonormal basis vectors
    basis - B, 3, 3

    out - B, 3, 3
    """
    try:
        u, s, v = torch.svd(basis)
    except:                     # torch.svd may have convergence issues for GPU and CPU.
        u, s, v = torch.svd(basis + 1e-3*basis.mean()*torch.rand_like(basis).type_as(basis))
    # u, s, v = torch.svd(basis)
    out = u @ v.transpose(-2, -1)    

    return out