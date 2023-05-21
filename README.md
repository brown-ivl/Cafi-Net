# Canonical Fields: Self-Supervised Learning of Pose-Canonicalized Neural Fields
[CVPR 2023 (Highlight)]
This is the official pytorch implementation of CaFi-Net as proposed in the paper [here](https://arxiv.org/abs/2212.02493).

## Abstract
Coordinate-based implicit neural networks, or neural fields, have emerged as useful representations of shape and appearance in 3D computer vision. Despite advances, however, it remains challenging to build neural fields for categories of objects without datasets like ShapeNet that provide "canonicalized" object instances that are consistently aligned for their 3D position and orientation (pose). We present Canonical Field Network (CaFi-Net), a self-supervised method to canonicalize the 3D pose of instances from an object category represented as neural fields, specifically neural radiance fields (NeRFs). CaFi-Net directly learns from continuous and noisy radiance fields using a Siamese network architecture that is designed to extract equivariant field features for category-level canonicalization. During inference, our method takes pre-trained neural radiance fields of novel object instances at arbitrary 3D pose and estimates a canonical field with consistent 3D pose across the entire category. Extensive experiments on a new dataset of 1300 NeRF models across 13 object categories show that our method matches or exceeds the performance of 3D point cloud-based methods.

## Dataset

## Citation
```
@InProceedings{agaram2023_cafinet,

author={Rohith Agaram and Shaurya Dewan and Rahul Sajnani and Adrien Poulenard and Madhava Krishna and Srinath Sridhar},

title={Canonical Fields: Self-Supervised Learning of Pose-Canonicalized Neural Fields},

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

month = {June}

year={2023}}
```
