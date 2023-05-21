import torch
import numpy as np
import sys
sys.path.append('../')
from utils.pointcloud_utils import sq_distance_mat,compute_centroids




def equilibrium_loss(unnormalized_capsules):
    a = torch.mean(unnormalized_capsules, dim=1, keepdims=False)
    am = torch.mean(a, dim=-1, keepdims=True)
    l = torch.subtract(a, am)
    l = l*l
    return torch.mean(l)
    
 
def localization_loss_new(points, capsules, centroids):


    points_centered = points[:, :, None] - centroids[:, None, :] # B, N, K, 3
    points_centered_activated = capsules[:, :, :, None] * points_centered

    l = points_centered.permute(0,2,1,3) # B, K, N, 3
    l_1 = points_centered_activated.permute(0,2,3,1) # B, K, 3, N

    covariance = l_1 @ l
    loss = torch.mean(torch.diagonal(covariance))
    return loss
    
    
def l2_distance_batch(x, y):
    z = x - y
    z = z * z
    z = torch.sum(z, dim=-1)
    z = torch.sqrt(z)
    return z

def chamfer_distance_l2_batch(X, Y, sq_dist_mat=None):

    if sq_dist_mat is None:

        # compute distance mat

        D2 = sq_distance_mat(X, Y)

    else:

        D2 = sq_dist_mat
    
    dXY = torch.sqrt(torch.max(torch.min(D2, dim=-1, keepdims=False).values, torch.tensor(0.000001)))

    dXY = torch.mean(dXY, dim=1, keepdims=False)

    dYX = torch.sqrt(torch.max(torch.min(D2, dim=1, keepdims=False).values, torch.tensor(0.000001)))

    dYX = torch.mean(dYX, dim=-1, keepdims=False)

    d = dXY + dYX

    return 0.5*d