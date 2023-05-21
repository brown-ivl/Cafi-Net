from canonicalization_metrics import *
import torch
import h5py
import os, sys, argparse
import numpy as np
sys.path.append("../")
from utils.losses import chamfer_distance_l2_batch, l2_distance_batch


if __name__=="__main__":

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Parser for generating frames")
    
    parser.add_argument("--path", type = str, required = True)
    parser.add_argument("--pc_path", type = str)
    parser.add_argument("--shape_idx", type=str, default = None)
    parser.add_argument("--rot_idx", type=str, default = None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--n_iter", default=20, type = int)
    parser.add_argument("--device", default = "cpu")
    
    
    args = parser.parse_args()
    ########################################################################


    AtlasNetClasses = ["car.h5"]

    if args.category is not None:
        print("single category")
        AtlasNetClasses = [args.category+".h5"]
    else:
        print("multi category")
    ma = 0.
    mb = 0.
    mc = 0.
    k = 0.
    shpe_idx_array = np.asarray([[ 8, 16, 12,  4, 14,  5, 10, 18, 19, 17,  3,  6,  9,  7,  1, 13,  0,  2,15, 11], [10,  1, 15, 13,  2, 19, 18,  7, 11,  5, 12,  6,  4, 17,  3, 16,  8,  0,9, 14],[19, 12, 14,  9,  6,  0,  5, 13, 17, 18,  2, 10, 15,  8,  7, 16,  3, 11,1,  4],[ 3, 19, 18, 14, 10, 12, 11, 13,  8,  2,  0,  7,  4,  6,  1, 15, 16, 17,9,  5],[ 8,  0, 18, 17,  2,  6,  5, 13, 10, 11,  1,  7, 12, 15, 19, 16,  9,  3,14,  4],[ 7, 18,  0, 15, 12,  5, 19,  8,  9, 17, 13, 11,  1,  4, 14, 10,  6, 16,3,  2],[11, 15,  4,  5, 14, 18,  3, 17,  6, 13, 12,  2, 10,  7, 16,  0,  9, 19,1,  8],[14,  4, 13, 11, 17, 16,  7, 10,  3,  8,  2, 18,  0, 19,  6, 12,  5,  1,15,  9],[ 3,  4,  0, 12, 14,  9, 10,  1,  6,  7,  2,  8, 17, 16, 13, 15, 19, 18,5, 11],[17, 18,  7, 16,  3, 13, 15,  0,  1,  2, 14, 11,  8, 19,  6, 12,  9, 10,4,  5]])
    args.pc_path = args.path

    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], args.path,args.pc_path,shapes_idx_path=args.shape_idx, batch_size=20, n_iter = args.n_iter, device = args.device)
        print("Category-Level Consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], args.path,args.pc_path,idx_path=args.rot_idx, batch_size=20, n_iter = args.n_iter, device = args.device)
        print("Instance-Level Consistency: ", b)
        mb += b
        c = class_consistency_metric_new(AtlasNetClasses[i], args.path,args.pc_path,shpe_idx_array,shapes_idx_path=args.shape_idx, batch_size=20, n_iter = args.n_iter, device = args.device)
        print("Ground Truth Equivariance Consistency: ", c)
	mc = mc + c
        k += 1.
