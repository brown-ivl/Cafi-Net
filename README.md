# Canonical Fields: Self-Supervised Learning of Pose-Canonicalized Neural Fields

*[Rohith Agaram](https://scholar.google.com/citations?user=Ni6qG7wAAAAJ), [Shaurya Dewan](https://scholar.google.com/citations?user=1FLYpxAAAAAJ), [Rahul Sajnani](https://scholar.google.com/citations?user=HAtfBjoAAAAJ&hl=en&oi=ao), [Adrien Poulenard](https://scholar.google.com/citations?hl=en&user=zsGbyGYAAAAJ), [Madhava Krishna](https://scholar.google.com/citations?user=QDuPGHwAAAAJ), [Srinath Sridhar](https://cs.brown.edu/people/ssrinath/)*

[CVPR 2023] This is the official PyTorch implementation of CaFi-Net as proposed in the paper [here](https://arxiv.org/abs/2212.02493).

https://github.com/brown-ivl/Cafi-Net/assets/56212873/00178eaa-acb4-4755-b1d6-ec95a532225a

## Abstract
Coordinate-based implicit neural networks, or neural fields, have emerged as useful representations of shape and appearance in 3D computer vision. Despite advances, however, it remains challenging to build neural fields for categories of objects without datasets like ShapeNet that provide "canonicalized" object instances that are consistently aligned for their 3D position and orientation (pose). We present Canonical Field Network (CaFi-Net), a self-supervised method to canonicalize the 3D pose of instances from an object category represented as neural fields, specifically neural radiance fields (NeRFs). CaFi-Net directly learns from continuous and noisy radiance fields using a Siamese network architecture that is designed to extract equivariant field features for category-level canonicalization. During inference, our method takes pre-trained neural radiance fields of novel object instances at arbitrary 3D pose and estimates a canonical field with consistent 3D pose across the entire category. Extensive experiments on a new dataset of 1300 NeRF models across 13 object categories show that our method matches or exceeds the performance of 3D point cloud-based methods.



## Dataset
Download the 1300 NeRF density fields dataset [here](https://nerf-fields.s3.amazonaws.com/final_fields/final_res_32.zip).
```
# Create base dataset directory
mkdir dataset
# Create directory for density fields dataset
mkdir dataset/nerf_fields
# Change directory
cd dataset/nerf_fields
# Download dataset
wget https://nerf-fields.s3.amazonaws.com/final_fields/final_res_32.zip
# Unzip dataset
unzip final_res_32.zip
```

## NeRF
Please find the PyTorch implementation of the NeRF codebase used for the generation of the dataset of 1300 NeRF density fields in the "nerf" folder. Instructions are provided in the README in the same folder.

## CaFi-Net
Please find the PyTorch implementation of CaFi-Net and its instructions in the "cafi_net" folder (./cafi_net). This implementation is still under development.

## Citation
```
@InProceedings{agaram2023_cafinet,
author={Rohith Agaram and 
        Shaurya Dewan and 
        Rahul Sajnani and 
        Adrien Poulenard and 
        Madhava Krishna and 
        Srinath Sridhar},
title={Canonical Fields: Self-Supervised Learning of Pose-Canonicalized Neural Fields},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June}
year={2023}}
```
