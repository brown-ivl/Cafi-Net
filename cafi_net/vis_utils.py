import seaborn as sns
import open3d as o3d
import numpy as np
import torch
import argparse, os, sys
import glob, os

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
    
    out = torch.squeeze(x, axis = 0).numpy()
    
    return out 


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
        # print(x.shape)
        label_map = np.ones((len(x), 3)) * 0.5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    pcd.colors = o3d.utility.Vector3dVector(label_map)

    o3d.io.write_point_cloud(filename, pcd)





def visualize_outputs(base_path, pointcloud_name, start = 0, max_num = 10, spacing = 1.0, skip = 1):
    '''
    Visualize point clouds in open3D 
    '''
    
    filename = glob.glob(os.path.join(base_path, "")  + pointcloud_name)
    filename.sort()
    print(filename)
    filename_2 = []
    for f in filename:
        #print(f)
        print(os.path.join(base_path, "") + f.split("/")[-1].split("_")[0] +"_" + pointcloud_name, f)
        #if (os.path.join(base_path, "") + f.split("/")[-1].split("_")[0] +"_" + pointcloud_name) == f:
        filename_2.append(f)
    filename = filename_2
    filename = filename[start::skip]
    filename = filename[:max_num]
    # print(filename)
    num_pcds = len(filename)
    if skip != 1:
        num_pcds = len(filename) // skip
    rows = np.floor(np.sqrt(num_pcds))
    pcd_list = []
    arrow_list = []
    pcd_iter = 0
    pcd_index = 0
    for pcd_file in filename:
        print(pcd_file)
        if pcd_iter % skip == 0:
        
            pcd = o3d.io.read_point_cloud(pcd_file)
            pcd.colors = pcd.points
            column_num = pcd_index // rows
            row_num = pcd_index % rows
            vector = (row_num * spacing, column_num * spacing, 0)
            #vector = [0, 0, 0]
            # print(vector)
            pcd.translate(vector)
            pcd_list.append(pcd)
            pcd_index += 1

            U, S, V = torch.pca_lowrank(torch.tensor(np.array(pcd.points)))
            arrow = get_arrow(torch.tensor([0, 0, 0]) + torch.tensor(list(vector)), V[:, 0] + torch.tensor(list(vector)))
            arrow_list.append(arrow)

        pcd_iter +=1


    #o3d.visualization.draw_geometries(arrow_list)
    o3d.visualization.draw_geometries(pcd_list)
     
def draw_geometries(pcds):
    """
    Draw Geometries
    Args:
        - pcds (): [pcd1,pcd2,...]
    """
    o3d.visualization.draw_geometries(pcds)

def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=0.2,
        cone_height=0.1,
        cylinder_radius=0.1,
        cylinder_height=0.5)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    scale = 1
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)


if __name__ == "__main__":
    

    # Argument parser
    parser = argparse.ArgumentParser(description = "Visualization script")
    parser.add_argument("--base_path", help = "Base path to folder", required = True)
    parser.add_argument("--pcd", help = "PCD string to visualize", required = True)
    parser.add_argument("--spacing", help = "Spacing", default = 2.0, type = float)
    parser.add_argument("--num_list", help = "indices", nargs="+", default = list(range(9)), type = int)
    parser.add_argument("--start", help = "start index", default = 0, type = int)
    parser.add_argument("--num", help = "number of models", default = 9, type = int)
    parser.add_argument("--skip", help = "number of models to skip", default = 1, type = int)


    args = parser.parse_args()
    #######################################################################
    
    
    visualize_outputs(args.base_path, args.pcd, spacing = args.spacing, start = args.start, max_num = args.num, skip = args.skip)
