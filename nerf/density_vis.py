import numpy as np
import open3d as o3d
import argparse
import os
from sklearn.cluster import KMeans


def cluster_sigmas(sigmas, n_clusters=2, power=1.0, exp=False, scale=1.0):
    print("Number of clusters = ", n_clusters)
    # dim, _, _ = sigmas.shape
    # sigmas = sigmas.reshape((-1, 1))
    # sigmas = sigmas + 1e2

    print("Sigmas range = ", np.min(sigmas), np.max(sigmas))
    relu_sigmas = np.where(sigmas > 0, sigmas, 0)
    powered_sigmas = relu_sigmas ** power
    print("Sigmas powered range = ", np.min(powered_sigmas), np.max(powered_sigmas))
    if exp:
        sigmas = 1. - np.exp(-scale * powered_sigmas)
    print("Sigmas final range = ", np.min(sigmas), np.max(sigmas))

    model = KMeans(init="k-means++", n_clusters=n_clusters)
    model.fit(sigmas)
    print("Cluster centers = ", model.cluster_centers_)

    labels = model.predict(sigmas)
    (clusters, counts) = np.unique(labels, return_counts=True)
    bg_label = clusters[np.where(counts == counts.max())[0]]
    clustered_sigmas = np.where(labels == bg_label, 0, 1)
    return clustered_sigmas
# .reshape((dim, dim, dim))


def visualize(sigmas_path, samples_path, sigma_thresh):
    sigmas = np.load(sigmas_path).reshape((-1, 1))
    samples = np.load(samples_path).reshape((-1, 3))

    # occ = np.where(sigmas > sigma_thresh)[0]
    # print("Thresholding with %f: Total = %d, Occupied = %d, Occupancy = %f" % (sigma_thresh, len(sigmas), len(occ), len(occ) / len(sigmas)))

    # thresh_pcd = o3d.geometry.PointCloud()
    # thresh_points = samples[np.where(sigmas > sigma_thresh)[0]]
    # thresh_pcd.points = o3d.utility.Vector3dVector(thresh_points)
    # o3d.visualization.draw_geometries([thresh_pcd])

    clustered_sigmas = cluster_sigmas(sigmas, 2, 2.0, True, 0.3109375)
    occ = np.where(clustered_sigmas != 0)[0]
    print("Clustering: Total = %d, Occupied = %d, Occupancy = %f" % (len(sigmas), len(occ), len(occ) / len(sigmas)))

    fg_pcd = o3d.geometry.PointCloud()
    fg_points = samples[np.where(clustered_sigmas != 0)[0]]
    fg_pcd.points = o3d.utility.Vector3dVector(fg_points)
    o3d.visualization.draw_geometries([fg_pcd])


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="NeRF density field visualization")
    parser.add_argument("--input", required=True, type = str)
    parser.add_argument("--res", default=32, type = int)
    parser.add_argument("--sigma_thresh", default=10.0, type = float)
    parser.add_argument("--max_files", default=10, type = int)
    args = parser.parse_args()

    count = 0
    for path, dirs, files in os.walk(args.input):
        for file in files:
            if "sigmas_%d.npy" % (args.res) not in file:
                continue

            print("Processing %s %s" % (path, file))
            sigmas_path = os.path.join(path, file)
            samples_path = os.path.join(path, file.replace("sigmas", "samples"))
            visualize(sigmas_path, samples_path, args.sigma_thresh)

            count += 1
            if count >= args.max_files:
                break
