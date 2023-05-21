import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import glob
import random
import matplotlib.pyplot as plt
import pickle
import h5py
from scipy.spatial.transform import Rotation as R


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius, cam_pose):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    device = c2w.device
    c2w = torch.tensor(c2w).to(device)
    return c2w


def read_pickle_file(path):
    objects = []
    with open(path, "rb") as fp:
        while True:
            try:
                obj = pickle.load(fp)
                objects.append(obj)

            except EOFError:
                break

    return objects 


def load_dataset(directory, canonical_pose = None, input_pose = None):
    cam_data_path = os.path.join(directory, "cam_data.pkl")
    cam_data = read_pickle_file(cam_data_path)[0]
    cams = {"width": 1280, "height": 720}

    imgs = {}
    image_dir = os.path.join(directory, "render/")
    images = glob.glob(image_dir + "**/*.png", recursive = True)
    images.sort()

    depth_dir = os.path.join(directory, "depth/")

    for i in range(len(images)):
        image_current = images[i]
        image_id = os.path.basename(image_current).split(".")[0]
        image_parent_dir = image_current.split("/")[-2]

        cam = cam_data[image_id]["K"]
        [cams["fx"], cams["fy"], cams["cx"], cams["cy"]] = cam
        c2w = cam_data[image_id]["extrinsics_opencv"]
        c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])
        c2w = np.linalg.inv(c2w)
        pose = c2w

        imgs[i] = {
            "camera_id": image_id,
            "t": pose[:3, 3].reshape(3, 1),
            "R": pose[:3, :3],
            "path": images[i],
            "pose": pose
        }

        imgs[i]["depth_path"] = os.path.join(depth_dir, "%s/%s_depth.npz" % (image_parent_dir, image_id))

    return imgs, cams

def main_loader(root_dir, scale, canonical_pose = None, input_pose = None):
    imgs, cams = load_dataset(root_dir, canonical_pose, input_pose)

    cams["fx"] = fx = cams["fx"] * scale
    cams["fy"] = fy = cams["fy"] * scale
    cams["cx"] = cx = cams["cx"] * scale
    cams["cy"] = cy = cams["cy"] * scale

    rand_key = random.choice(list(imgs))
    test_img = cv2.imread(imgs[rand_key]["path"])
    h, w = test_img.shape[:2]
    cams["height"] = round(h * scale)
    cams["width"] = round(w * scale)

    cams["intrinsic_mat"] = np.array([
        [fx, 0, cx],
        [0, -fy, cy],
        [0, 0, -1]
        ])

    return imgs, cams 


def load_brics_data(basedir, res = 1, skip = 1, max_ind = 54, canonical_pose = None, input_pose = None, num_poses = 300):
    imgs, cams = main_loader(basedir, res, canonical_pose, input_pose)
    all_ids = []
    all_imgs = []
    all_poses = []
    all_depths = []

    for index in range(0, max_ind, skip):
        all_ids.append(imgs[index]["camera_id"])

        n_image = imageio.imread(imgs[index]["path"]) / 255.0
        h, w = n_image.shape[:2]
        resized_h = round(h * res)
        resized_w = round(w * res)
        n_image = cv2.resize(n_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        all_imgs.append(n_image)

        n_pose = imgs[index]["pose"]
        all_poses.append(n_pose)
        
        n_depth = np.load(imgs[index]["depth_path"])['arr_0']
        n_depth = np.where(n_depth == np.inf, 0, n_depth)
        n_depth = np.where(n_depth > 100, 0, n_depth)
        n_depth = cv2.resize(n_depth, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        all_depths.append(n_depth)
    
    all_poses = np.array(all_poses)
    all_imgs = np.array(all_imgs).astype(np.float32)
    all_depths = np.array(all_depths).astype(np.float32)

    c2w = all_poses[all_ids.index("left_5"), :, :]
    input_pose_4 = np.identity(4)
    canonical_pose_4 = np.identity(4)
    transform = False
    canonical_poses = []

    if input_pose is not None:
        input_pose_4[:3, :3] = input_pose
        transform = True

    if canonical_pose is not None:
        canonical_pose_4[:3, :3] = canonical_pose
        transform = True

    if transform:
        t = np.array([0.0, -0.5, 4.5]).T  #known from the brics simulator
        nerf_w_2_transform_w = np.identity(4)
        nerf_w_2_transform_w[:3, -1] = -t
        temp = nerf_w_2_transform_w @ c2w 

        for i in range(num_poses):
            angle = np.linspace(0, 360, num_poses)[i]
            circular_pose = pose_spherical(0.0, angle, 0.0, c2w).cpu().numpy() 
            pose = np.linalg.inv(canonical_pose_4 @ input_pose_4) @ np.linalg.inv(circular_pose) @ temp
            # pose = np.linalg.inv(canonical_pose_4) @ np.linalg.inv(circular_pose) @ temp
            pose[:3, -1] += t
            canonical_poses.append(pose)

        all_poses = np.array(canonical_poses)

    i_val = []
    sides = ["back", "bottom", "front", "left", "right", "top"]
    for side_idx in range(len(sides)):
        panel_idx = np.random.randint(1, 10) 
        val_camera_id = "%s_%d" % (sides[side_idx], panel_idx)
        val_idx = all_ids.index(val_camera_id)
        i_val.append(val_idx)

    indices = np.arange(len(all_imgs))
    i_train = np.array(list(set(indices).difference(set(i_val))))
    i_test = i_val
    i_split = [i_train, i_val, i_test]

    render_poses = np.array(all_poses)

    return all_imgs, all_poses, render_poses, cams, all_depths, i_split

