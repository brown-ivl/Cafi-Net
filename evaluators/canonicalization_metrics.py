import torch
import h5py
import os, sys
import numpy as np
sys.path.append("../")
from utils.losses import chamfer_distance_l2_batch, l2_distance_batch
import open3d as o3d
from pytorch3d.loss import chamfer_distance
import open3d as o3d
distance_metric = chamfer_distance_l2_batch

# distance_metric = l2_distance_batch

def orient(r):
    """
    shape = list(r.shape)
    shape = shape[:-2]
    _, u, v = tf.linalg.svd(r)

    R = tf.einsum('bij,bkj->bik', u, v)



    s = tf.stack([tf.ones(shape), tf.ones(shape), tf.sign(tf.linalg.det(R))], axis=-1)
    # u = tf.einsum('bj,bij->bij', s, u)
    u = tf.multiply(tf.expand_dims(s, axis=-1), u)
    # v = tf.multiply(tf.expand_dims(s, axis=1), v)
    R = tf.einsum('bij,bkj->bik', u, v)
    """

    return r

def save_h5_(h5_filename, data,data_dtype='float32'):
    h5_fout = h5py.File(h5_filename,"w")

    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_fout.close()
        
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def save_h5(h5_filename, data, normals=None, subsamplings_idx=None, part_label=None,
            class_label=None, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)

    h5_fout.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)

    if normals is not None:
        h5_fout.create_dataset(
            'normal', data=normals,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)

    if subsamplings_idx is not None:
        for i in range(len(subsamplings_idx)):
            name = 'sub_idx_' + str(subsamplings_idx[i].shape[1])
            h5_fout.create_dataset(
                name, data=subsamplings_idx[i],
                compression='gzip', compression_opts=1,
                dtype='int32')

    if part_label is not None:
        h5_fout.create_dataset(
            'pid', data=part_label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)

    if class_label is not None:
        h5_fout.create_dataset(
            'label', data=class_label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def torch_random_rotation(shape):
    if isinstance(shape, int):
        shape = [shape]

    batch_size = shape[0]
    t = torch.random(shape + [3])
    c1 = torch.cos(2 * np.pi * t[:, 0])
    s1 = torch.sin(2 * np.pi * t[:, 0])

    c2 = torch.cos(2 * np.pi * t[:, 1])
    s2 = torch.sin(2 * np.pi * t[:, 1])

    z = torch.zeros(shape)
    o = torch.ones(shape)

    R = torch.stack([c1, s1, z, -s1, c1, z, z, z, o], dim=-1)
    R = torch.reshape(R, shape + [3, 3])

    v1 = torch.sqrt(t[:, -1])
    v3 = torch.sqrt(1-t[:, -1])
    v = torch.stack([c2 * v1, s2 * v1, v3], dim=-1)
    H = torch.tile(torch.unsqueeze(torch.eye(3), 0), (batch_size, 1, 1)) - 2.* torch.einsum('bi,bj->bij', v, v)
    M = -torch.einsum('bij,bjk->bik', H, R)
    return M


def batch_of_frames(n_frames, filename, path):

    I = torch.unsqueeze(torch.eye(3),0)
    R = torch_random_rotation(n_frames - 1)

    R = torch.cat([I, R], dim=0)
    print("R shape")
    print(R.shape)
    print(R)

    h5_fout = h5py.File(os.path.join(path, filename), 'w')
    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()


# batch_of_frames(n_frames=128, filename="rotations.h5", path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024")



AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]

def save_rotation(h5_filename, src_path, tar_path, rots_per_shape=512, batch_size=512):
    filename = os.path.join(src_path, h5_filename)
    f = h5py.File(filename)
    print(filename)
    data = f['data'][:]
    num_shapes = data.shape[0]
    num_batches = num_shapes // batch_size
    residual = num_shapes % batch_size
    R = []

    """
    if num_batches == 0:
        batch = tf_random_rotation(rots_per_shape * num_shapes)
        batch = tf.reshape(batch, (-1, rots_per_shape, 3, 3))
        R.append(np.asarray(batch, dtype=np.float))
    """

    for i in range(num_batches):
        a = i*batch_size
        b = min((i+1)*batch_size, num_shapes)
        if a < b:
            batch = torch_random_rotation((b - a)*rots_per_shape)
            batch = torch.reshape(batch, (-1, rots_per_shape, 3, 3))
            batch = np.asarray(batch, dtype=np.float)
            R.append(batch)


    if residual > 0:
        batch = torch_random_rotation(residual * rots_per_shape)
        batch = torch.reshape(batch, (-1, rots_per_shape, 3, 3))
        batch = np.asarray(batch, dtype=np.float)
        R.append(batch)



    # R = tf.concat(R, axis=0)
    R = np.concatenate(R, axis=0)
    print(data.shape)
    print(R.shape)

    # R = np.asarray(R, dtype=np.float)


    h5_fout = h5py.File(os.path.join(tar_path, h5_filename), 'w')

    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()


"""
AtlasNetPath = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024"

for name in AtlasNetClasses:
    save_rotation(name, os.path.join(AtlasNetPath, 'valid'), os.path.join(AtlasNetPath, 'rotations_valid'))
    save_rotation(name, os.path.join(AtlasNetPath, 'train'), os.path.join(AtlasNetPath, 'rotations_train'))


exit(666)
"""

def mean(x, batch_size=512):
    num_shapes = x.shape[0]
    num_batches = num_shapes // batch_size
    remainder = num_shapes // batch_size
    m = []
    k = 0.


    for i in range(num_batches):
        a = i * batch_size
        b = min((i + 1) * batch_size, num_shapes)
        if a < b:
            k += float(b - a)
            batch = x[a:b, ...]
            m.append(torch.sum(batch, dim=0, keepdims=True))

    if remainder > 0:
        a = num_batches * batch_size
        b = num_shapes
        if a < b:
            k += float(b - a)
            batch = x[a:b, ...]
            m.append(torch.sum(batch, dim=0, keepdims=True))

    m = torch.cat(m, dim=0)
    m = torch.sum(m, dim=0, keepdims=False)
    m /= k
    return m

def var(x, batch_size=512):
    num_shapes = x.shape[0]
    num_batches = num_shapes // batch_size
    remainder = num_shapes // batch_size
    v = []
    k = 0.
    m = torch.unsqueeze(mean(x, batch_size=512), dim=0)


    for i in range(num_batches):
        a = i * batch_size
        b = min((i + 1) * batch_size, num_shapes)
        if a < b:
            k += float(b - a)
            xi = x[a:b, ...]

            vi = torch.sub(xi, m)
            vi = vi * vi
            vi = torch.sum(vi)
            v.append(vi)

    if remainder > 0:
        a = num_batches * batch_size
        b = num_shapes
        if a < b:
            k += float(b - a)
            xi = x[a:b, ...]
            vi = torch.sub(xi, m)
            vi = vi * vi
            vi = torch.sum(vi)
            v.append(vi)

    v = torch.stack(v, dim=0)
    v = torch.sum(v)
    v /= k
    return v

def std(x, batch_size=512):
    return torch.sqrt(var(x, batch_size))

def sq_dist_mat(x, y):
    r0 = torch.mul(x, x)
    r0 = torch.sum(r0, axis=2, keepdims=True)

    r1 = torch.mul(y, y)
    r1 = torch.sum(r1, axis=2, keepdims=True)
    r1 = torch.permute(r1,(0, 2, 1))

    sq_distance_mat = r0 - 2. * torch.matmul(x, y.permute(0, 2, 1)) + r1
    return sq_distance_mat


def var_(x, axis_mean=0, axis_norm=1):
    mean = torch.mean(x, dim=axis_mean, keepdims=True)
    y = torch.sub(x, mean)
    yn = torch.sum(y * y, dim=axis_norm, keepdims=False)
    yn = torch.mean(yn, dim=axis_mean, keepdims=False)
    return yn, mean

def std_(x, axis_mean=0, axis_norm=1):
    yn, mean = var_(x, axis_mean=axis_mean, axis_norm=axis_norm)
    return torch.sqrt(yn), mean


def pca_align(x):
    c = torch.mean(x, dim=1, keepdims=True)
    centred_x = torch.sub(x, c)
    covar_mat = torch.mean(torch.einsum('bvi,bvj->bvij', centred_x, centred_x), dim=1, keepdims=False)
    _, v = np.linalg.eigh(covar_mat.detach().numpy())
    v = torch.from_numpy(v)
    
    x = torch.einsum('bij,bvi->bvj', v, centred_x)
    return x, v.permute(0,2,1)

def visualize_outputs(data, start = 0, max_num = 10, spacing = 2.0, skip = 1):
    '''
    Visualize point clouds in open3D 
    '''
    
    
    num_pcds = 10
    
    rows = np.floor(np.sqrt(num_pcds))
    pcd_list = []
    arrow_list = []
    pcd_iter = 0
    pcd_index = 0
    for _ in range(num_pcds):
        pts = data[_].cpu().detach().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = pcd.points
        column_num = pcd_index // rows
        row_num = pcd_index % rows
        vector = (row_num * spacing, column_num * spacing, 0)
        pcd.translate(vector)
        pcd_list.append(pcd)
        pcd_index += 1

        
    
    o3d.visualization.draw_geometries(pcd_list)
    

    
    
def save_pca_frames_(filename, x_path, batch_size=20, num_rots=128):
   
    
    obj_name = filename.split('.')[0]
    x = loadh5(os.path.join(x_path, filename))
    x_data = x.copy()
    r = loadh5(os.path.join(x_path, obj_name +"_rotations.h5"))
    x = torch.from_numpy(x)
    r = torch.from_numpy(r)
    num_shapes = x.shape[0]
    x = extend_(x, batch_size)
    
    save_directory = x_path + "pca"
    
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    # r = extend_(r, batch_size)
    
    h5_filename = save_directory + "/"+ obj_name +".h5"
    save_h5_(h5_filename,x_data)
    
    h5_filename = save_directory + "/"+ obj_name +"_rotations.h5"
    save_h5_(h5_filename, r.cpu().detach().numpy())
    
    num_batches = x.shape[0] // batch_size
    R = []

    for i in range(num_batches):
        print(100.*i/num_batches)
        xi = x[i * batch_size:(i + 1) * batch_size, ...]

        Ri = []
        canonical_frame = []
        data_ = []
        input_data= []
        for j in range(num_rots):
            xij = torch.einsum("ij,bvj->bvi", r[j, ...], xi)
            # yij = pca_frame(xij)
            data,frame = pca_align(xij)
            Ri.append(data)
            if j%10 == 0:
                data_.append(data[0])
                input_data.append(xij[0])
            
                
            canonical_frame.append(frame)
            
        
        #data_ = torch.stack(data_,axis=0)
        #input_data = torch.stack(input_data,axis=0)
        #visualize_outputs(data_)  
        #visualize_outputs(input_data)          
        Ri = torch.stack(Ri, axis=1)
        R.append(np.asarray(Ri, dtype=np.float))
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]
    
    
        
    canonical_frame = torch.stack(canonical_frame,axis=0).permute(1,0,2,3)
    h5_filename = save_directory + "/"+ obj_name +"_canonical.h5"
    save_h5_(h5_filename, canonical_frame.cpu().detach().numpy())
    
    
    '''
    filename_ = obj_name + "_pca.h5"
    h5_fout = h5py.File(os.path.join(x_path + "pca", filename), 'w')
    h5_fout.create_dataset(
        'data', data=R,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.close()'''


def normalize(x):
    s, m = std_(x, axis_mean=1, axis_norm=-1)
    x = torch.div(torch.sub(x, m), s)
    return x


def orth_procrustes(x, y):
    x = normalize(x)
    y = normalize(y)
    xty = torch.einsum('bvi,bvj->bij', y, x)
    s, u, v = torch.linalg.svd(xty)
    r = torch.einsum('bij,bkj->bik', u, v)
    return r


def extend_(x, batch_size):
    last_batch = x.shape[0] % batch_size
    if last_batch > 0:
        X_append = []
        for i in range(batch_size - last_batch):
            X_append.append(x[i, ...])
        X_append = torch.stack(X_append, dim=0)
        y = torch.cat([x, X_append], dim=0)
    else:
        y = x
    return y

def xyz2yzx(x):
    return torch.stack([x[..., 1], x[..., 2], x[..., 0]], dim=-1)

def yzx2xyz(x):
    return torch.stack([x[..., 2], x[..., 0], x[..., 1]], dim=-1)

def yzx2xyzConj(R):
    R = yzx2xyz(torch.linalg.matrix_transpose(R))
    R = torch.linalg.matrix_transpose(R)
    # return xyz2yzx(R)
    return R



def class_consistency_frames(r, r_can, batch_size=32):
    
    num_batches = r.shape[0] // batch_size
    num_shapes = r.shape[0]
    num_rots = min(r.shape[1], r_can.shape[1])
    r = r[:, :num_rots, ...]
    r_can = r_can[:, :num_rots, ...]
    r = extend_(r, batch_size)
    r_can = extend_(r_can, batch_size)
    
    


    R = []
    for i in range(num_batches):
        a = i * batch_size
        b = (i + 1) * batch_size
        rj = r[a:b, ...]
        r_can_j = r_can[a:b, ...]
        # Ri = tf.matmul(r_can_j, rj, transpose_a=True)
        #  Ri = tf.matmul(r_can_j, rj, transpose_b=True)
        # Ri = tf.matmul(r_can_j, r_can_j, transpose_b=True)
        Ri = r_can_j
        Ri = orient(Ri)
        # Ri = tf.matmul(r_can_j, rj)
        # Ri = np.stack(np.asarray(Ri, dtype=np.float32), axis=1)
        R.append(np.asarray(Ri, dtype=np.float32))
    R = np.concatenate(R, axis=0)
    R = R[:num_shapes, ...]
    # print(R)
    return R



def visualize_method(filename, x_path,):
    obj_name = filename.split('.')[0]
    
    x = loadh5(os.path.join(x_path, filename))
    r_can = loadh5(os.path.join(x_path, obj_name +"_canonical.h5"))
    r_input = loadh5(os.path.join(x_path, obj_name +"_rotations.h5"))
    x = torch.from_numpy(x)
    r_can = torch.from_numpy(r_can)
    r_input = torch.from_numpy(r_input)
    pcds_ = []
    pcds_1 = []
    for i in range(x.shape[0]):
        for j in range(r_input.shape[0]):
            rj = r_input[j, ...]
            xij = torch.einsum("ij,bvj->bvi", rj, x)
            y0i = torch.einsum("bij,bvj->bvi", orient(r_can[:,j,:]), xij)
            if j%10 == 0:
               pcds_.append(y0i[4]) 
               pcds_1.append(xij[4]) 
               
    input_data = torch.stack(pcds_,axis=0)
    input_data_1 = torch.stack(pcds_1,axis=0)
    visualize_outputs(input_data)  
    visualize_outputs(input_data_1)  

def class_consistency_metric_(x, r_input, r_can, val_points,idx=None, batch_size=32):
    
    num_rots = r_input.shape[0]
    n_shapes = x.shape[0]
    if idx is None:
        idx = torch.randperm(n_shapes)


    r_can_0 = r_can
    r_can_1 = r_can[idx]
    x = extend_(x, batch_size)
    r_can_0 = extend_(r_can_0, batch_size)
    r_can_1 = extend_(r_can_1, batch_size)
    num_batches = x.shape[0] // batch_size
    D = []
    for j in range(num_rots):
        rj = r_input[j, ...]
        d = []
        d_ = []
        for i in range(num_batches):
            r_can_0_ij = r_can_0[i * batch_size:(i + 1) * batch_size, j, ...]
            r_can_1_ij = r_can_1[i * batch_size:(i + 1) * batch_size, j, ...]
            
            xi = x[i * batch_size:(i + 1) * batch_size, ...]
            
            #points_val = val_points[i * batch_size:(i + 1) * batch_size, ...]
            points_val =[1024]*batch_size
            xij = torch.einsum("ij,bvj->bvi", rj, xi)
            y0i = torch.einsum("bij,bvj->bvi", orient(r_can_0_ij), xij)
            y1i = torch.einsum("bij,bvj->bvi", orient(r_can_1_ij), xij)
            for _ in range(y0i.shape[0]):
                d_.append(float(chamfer_distance(y0i[_][:int(points_val[_])].unsqueeze(0),y1i[_][:int(points_val[_])].unsqueeze(0))[0]))
        
       
        #d = np.concatenate(d, axis=0)
        d = np.asarray(d_)
        d = d[:n_shapes, ...]
        
        D.append(np.mean(d))
    D = np.stack(D, axis=0)
    D = np.mean(D)
    return float(D)

def class_consistency_metric_new_(x, r_input, r_can,val_points=None, idx=None, batch_size=32):
    num_rots = r_input.shape[0]
    n_shapes = x.shape[0]
    if idx is None:
        idx = torch.randperm(n_shapes)
        
    rot_idx = torch.randperm(num_rots)
    r_input_0 = r_input
    r_input_1  = r_input[rot_idx]


    r_can_0 = r_can
    r_can_1 = r_can[idx]
    r_can_1 = r_can_1[:,rot_idx,...]
    
            
    x = extend_(x, batch_size)
    r_can_0 = extend_(r_can_0, batch_size)
    r_can_1 = extend_(r_can_1, batch_size)
    num_batches = x.shape[0] // batch_size
    D = []
    for j in range(num_rots):
        rj0 = r_input_0[j, ...]
        rj1 = r_input_1[j, ...]
        d = []
        d_ = []
        for i in range(num_batches):
            r_can_0_ij = r_can_0[i * batch_size:(i + 1) * batch_size, j, ...]
            r_can_1_ij = r_can_1[i * batch_size:(i + 1) * batch_size, j, ...]
            
            xi = x[i * batch_size:(i + 1) * batch_size, ...]
            #points_val = val_points[i * batch_size:(i + 1) * batch_size, ...]
            points_val =[1024]*batch_size
            xij0 = torch.einsum("ij,bvj->bvi", rj0, xi)
            xij1 = torch.einsum("ij,bvj->bvi", rj1, xi)
            
            y0i = torch.einsum("bij,bvj->bvi", orient(r_can_0_ij), xij0)
            y1i = torch.einsum("bij,bvj->bvi", orient(r_can_1_ij), xij1)

        
            for _ in range(y0i.shape[0]):
                d_.append(float(chamfer_distance(y0i[_][:int(points_val[_])].unsqueeze(0),y1i[_][:int(points_val[_])].unsqueeze(0))[0]))
        #d = np.concatenate(d, axis=0)
        d = np.asarray(d_)
        d = d[:n_shapes, ...]
        
        D.append(np.mean(d))
    D = np.stack(D, axis=0)
    D = np.mean(D)
    return float(D)


def loadh5(path):
    fx_input = h5py.File(path, 'r')
    x = fx_input['data'][:]
    fx_input.close()
    return x
    
def load_file_names(path,name):
    filename = glob.glob(os.path.join(path, "")  + name)
    filename.sort()
    file_list = []
    for f in filename:
        print(os.path.join(base_path, "") + f.split("/")[-1].split("_")[0] +"_" + name, f)
        file_list.append(f)
        
    return file_list
    


def class_consistency_metric_new(filename, x_path, pc_path,shape_idx_array, shapes_idx_path=None, batch_size=32, n_iter=10, device = "cpu"):
    
    
    obj_name = filename.split('.')[0]
    
    x = loadh5(os.path.join(pc_path, obj_name +".h5"))
    r_can = loadh5(os.path.join(x_path, obj_name +"_canonical.h5"))
    r_input = loadh5(os.path.join(x_path, obj_name +"_rotations.h5"))
    #val_points = loadh5(os.path.join(pc_path, obj_name +"_val_points.h5"))
    val_points = None

    x = torch.from_numpy(x).to(device)
    x = (x - torch.mean(x , axis=1,keepdim=True))
    
    r_can = torch.from_numpy(r_can).to(device)
    r_input = torch.from_numpy(r_input).to(device)
    m = 0.
    
    if shapes_idx_path is not None:
        idx = loadh5(os.path.join(shapes_idx_path, filename))
        idx = torch.from_numpy(idx).to(torch.int64).to(device)
        n_iter = min(n_iter, idx.shape[0])
        for i in range(n_iter):
            m += class_consistency_metric_new_(x, r_input,
                                           r_can,val_points,idx[i, ...], batch_size)
    else:
        idx = None
        for i in range(n_iter):
            m += class_consistency_metric_new_(x, r_input, r_can,val_points, idx, batch_size)
    return m / n_iter
    
    
def class_consistency_metric(filename, x_path, pc_path,shapes_idx_path=None, batch_size=32, n_iter=10, device = "cpu"):
    

    
    obj_name = filename.split('.')[0]
    
    x = loadh5(os.path.join(pc_path, obj_name +".h5"))
    r_can = loadh5(os.path.join(x_path, obj_name +"_canonical.h5"))
    r_input = loadh5(os.path.join(x_path, obj_name +"_rotations.h5"))
    #val_points = loadh5(os.path.join(pc_path, obj_name +"_val_points.h5"))
    val_points = None

    x = torch.from_numpy(x).to(device)
    x = (x - torch.mean(x , axis=1,keepdim=True))
    r_can = torch.from_numpy(r_can).to(device)
    r_input = torch.from_numpy(r_input).to(device)
    m = 0.
    
    if shapes_idx_path is not None:
        idx = loadh5(os.path.join(shapes_idx_path, filename))
        idx = torch.from_numpy(idx).to(torch.int64).to(device)
        n_iter = min(n_iter, idx.shape[0])
        for i in range(n_iter):
            m += class_consistency_metric_(x, r_input,
                                           r_can,val_points,idx[i, ...], batch_size)
    else:
        idx = None
        for i in range(n_iter):
            m += class_consistency_metric_(x, r_input, r_can, val_points,idx, batch_size)
    return m / n_iter


def equivariance_metric_(x, r_input, r_can, val_points=None, batch_size=20, idx=None):

    num_shapes = x.shape[0]
    num_rots = r_input.shape[0]
    if idx is None:
        idx = torch.randperm(num_rots)

    r_can_0   = r_can
    r_can_1   = r_can[:,idx]
    r_input_0 = r_input
    r_input_1 = r_input[idx]
    
    x = extend_(x, batch_size)
    
    r_can_0 = extend_(r_can_0, batch_size)
    r_can_1 = extend_(r_can_1, batch_size)
    # r_input_0 = extend_(r_input_0, batch_size)
    # r_input_1 = extend_(r_input_1, batch_size)

    num_batches = x.shape[0] // batch_size
    D = []
    for i in range(num_batches):
        d = []
        d_ = []
        for j in range(num_rots):
            r0j = r_input_0[j, ...]
            r1j = r_input_1[j, ...]
            r_can_0_ij = r_can_0[i * batch_size:(i + 1) * batch_size, j, ...]
            r_can_1_ij = r_can_1[i * batch_size:(i + 1) * batch_size, j, ...]
            
            xi = x[i * batch_size:(i + 1) * batch_size, ...]
            x0ij = torch.einsum("ij,bvj->bvi", r0j, xi)
            y0i = torch.einsum("bij,bvj->bvi", orient(r_can_0_ij), x0ij)
            x1ij = torch.einsum("ij,bvj->bvi", r1j, xi)
            y1i = torch.einsum("bij,bvj->bvi", orient(r_can_1_ij), x1ij)
            #points_val = val_points[i * batch_size:(i + 1) * batch_size, ...]
            points_val = [1024]*batch_size
            d_int = []
            for _ in range(y0i.shape[0]):
                d_int.append(float(chamfer_distance(y0i[_][:int(points_val[_])].unsqueeze(0),y1i[_][:int(points_val[_])].unsqueeze(0))[0]))
            d_.append(d_int)
        
        #d = np.stack(d, axis=1)
        d = np.asarray(d_)
        d = np.mean(d, axis=1, keepdims=False)
        D.append(d)
    D = np.concatenate(D, axis=0)
    D = D[:num_shapes, ...]
    D = np.mean(D)
    return float(D)


def equivariance_metric(filename, x_path, pc_path,batch_size, idx_path=None, n_iter=10, device = "cpu"):
    
    obj_name = filename.split('.')[0]
    
    x = loadh5(os.path.join(pc_path, obj_name +".h5"))
    r_can = loadh5(os.path.join(x_path, obj_name +"_canonical.h5"))
    r_input = loadh5(os.path.join(x_path, obj_name +"_rotations.h5"))
    #val_points = loadh5(os.path.join(pc_path, obj_name +"_val_points.h5"))
    val_points = None
    x = torch.from_numpy(x).to(device)
    x = (x - torch.mean(x , axis=1,keepdim=True))
    r_can = torch.from_numpy(r_can).to(device)
    r_input = torch.from_numpy(r_input).to(device)
    spacing = 1.0
    '''
    for _ in range(r_can.shape[1]):
        #random_rot_pcd = torch.matmul(r_input,x.permute(1,0)).permute(0,2,1)
        random_rot_pcd = torch.matmul(r_input[0],x.permute(0,2,1)).permute(0,2,1)
        can_pcd        = torch.matmul(r_can[:,_],random_rot_pcd.permute(0,2,1)).permute(0,2,1)
        can_pcd_numpy  = can_pcd.detach().numpy()
        
        num_pcds = can_pcd_numpy.shape[0]
        rows = np.floor(np.sqrt(num_pcds))
        pcd_list = []
        arrow_list = []
        pcd_iter = 0
        pcd_index = 0
        for k in range(num_pcds):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(can_pcd_numpy[k])
            pcd.colors = pcd.points
            column_num = pcd_index // rows
            row_num = pcd_index % rows
            vector = (row_num * spacing, column_num * spacing, 0)
            pcd.translate(vector)
            pcd_list.append(pcd)
            pcd_index += 1

        #o3d.visualization.draw_geometries(pcd_list)'''
    
    m = 0.
    if idx_path is None:
        idx = None
        for i in range(n_iter):
            m += equivariance_metric_(x, r_input, r_can,val_points,batch_size, idx=idx)
    else:
        idx = loadh5(idx_path)
        idx = torch.from_numpy(idx).to(torch.int64)     
        n_iter = min(n_iter, idx.shape[0])
        for i in range(n_iter):
            m += equivariance_metric_(x, r_input, r_can, batch_size, idx=idx[i, ...])
    return m / n_iter


def class_consistency_umetric_(x, r_input, r_can, val_points=None,idx_shapes=None, idx_rots=None, batch_size=32):

  
    num_rots = min(r_input.shape[0], r_can.shape[0])
    n_shapes = x.shape[0]
    if idx_shapes is None:
        idx_shapes = torch.randperm(n_shapes)
    if idx_rots is None:
        idx_rots = torch.randperm(num_rots)
    else:
        idx_rots = idx_rots[:num_rots, ...]

    r_can_0 = r_can
    r_can_1 = r_can[:,idx_rots]
    r_can_1 = r_can_1[idx_shapes]
    r_input_0 = r_input
    r_input_1 = r_input[idx_rots]
    x_0 = x
    x_1 = x[idx_shapes]
    #val_points_0 = val_points
    #val_points_1 = val_points[idx_shapes]
    
    val_points_0 = [1024]*batch_size
    val_points_1 = [1024]*batch_size
    
    x_0 = extend_(x_0, batch_size)
    x_1 = extend_(x_1, batch_size)
    r_can_0 = extend_(r_can_0, batch_size)
    r_can_1 = extend_(r_can_1, batch_size)

    num_batches = x.shape[0] // batch_size
    D = []
    d_ = []
    for j in range(num_rots):
        r0j = r_input_0[j, ...]
        r1j = r_input_1[j, ...]
        d = []
        for i in range(num_batches):
            r_can_0_ij = r_can_0[i * batch_size:(i + 1) * batch_size, j, ...]
            r_can_1_ij = r_can_1[i * batch_size:(i + 1) * batch_size, j, ...]
            x0i = x_0[i * batch_size:(i + 1) * batch_size, ...]
            x1i = x_1[i * batch_size:(i + 1) * batch_size, ...]
            x0ij = torch.einsum("ij,bvj->bvi", r0j, x0i)
            x1ij = torch.einsum("ij,bvj->bvi", r1j, x1i)
            y0i = torch.einsum("bij,bvj->bvi", orient(r_can_0_ij), x0ij)
            y1i = torch.einsum("bij,bvj->bvi", orient(r_can_1_ij), x1ij)
            d_int = []
            for _ in range(y0i.shape[0]):
                d_int.append(float(chamfer_distance(y0i[_][:int(val_points_0[_])].unsqueeze(0),y1i[_][:int(val_points_1[_])].unsqueeze(0))[0]))
            d_.append(d_int)
            
       
        
        d = np.asarray(d_)
        d = d[:n_shapes, ...]
        D.append(np.mean(d))
    D = np.stack(D, axis=0)
    D = np.mean(D)
    return float(D)

def class_consistency_umetric(filename, x_path,pc_path, idx_shapes_path=None, idx_rots_path=None, batch_size=32, n_iter=10, device = "cpu"):
    
    obj_name = filename.split('.')[0]
    
    x = loadh5(os.path.join(pc_path, obj_name +".h5"))
    r_can = loadh5(os.path.join(x_path, obj_name +"_canonical.h5"))
    r_input = loadh5(os.path.join(x_path, obj_name +"_rotations.h5"))
    #val_points = loadh5(os.path.join(pc_path, obj_name +"_val_points.h5"))
    val_points = None
    x = torch.from_numpy(x).to(device)
    r_can = torch.from_numpy(r_can).to(device)
    r_input = torch.from_numpy(r_input).to(device)
    x = (x - torch.mean(x , axis=1,keepdim=True))
    if idx_shapes_path is not None:
        idx_shapes = loadh5(os.path.join(idx_shapes_path, filename))
        idx_shapes = torch.from_numpy(idx_shapes).to(torch.int64)
        n_iter = min(n_iter, idx_shapes.shape[0])
    else:
        idx_shapes = None

    if idx_rots_path is not None:
        idx_rots = loadh5(idx_rots_path)
        idx_rots = torch.from_numpy(idx_rots).to(torch.int64)
        n_iter = min(n_iter, idx_rots.shape[0])
    else:
        idx_rots = None

    m = 0.
    for i in range(n_iter):
        ri = None
        si = None
        if idx_rots is not None:
            ri = idx_rots[i, ...]
        if idx_shapes is not None:
            si = idx_shapes[i, ...]

        m += class_consistency_umetric_(x, r_input, r_can,val_points,
                                        idx_shapes=si, idx_rots=ri, batch_size=batch_size)

    return m / n_iter


def icp_class_consistency_metric(x, batch_size=32, n_shuffles=10, n_iter=5):
    """
    :param x: canonicalized shapes (num_shapes, num_points, 3)
    :param batch_size:
    :param n_shuffles: number of times we shuffle x for self comparison
    :param n_iter: number of icp iterations
    :return:
    """
    b = x.shape[0]
    u_ = b % batch_size
    n = b // batch_size
    var_ = 0.
    m = torch.unsqueeze(torch.reshape(torch.eye(3), (9,)), dim=1)
    for j in range(n_shuffles):
        idx = np.random.permutation(x.shape[0])
        y_ = np.take(x, indices=idx, axis=0)
        k = 0.
        varj = 0.
        for i in range(n):
            k += 1.
            r = icp(x[i * batch_size:(i + 1) * batch_size, ...], y_[i * batch_size:(i + 1) * batch_size, ...], n_iter=n_iter)
            r = torch.reshape(r, (r.shape[0], -1))
            r_m = torch.sub(r, m)
            r_m = r_m * r_m
            rn = torch.sum(r_m, dim=-1)
            varj += float(torch.mean(rn))

        if u_ > 0:
            k += u_ / float(batch_size)
            r = icp(x[n * batch_size:, ...], y_[n * batch_size:, ...], n_iter=n_iter)
            r = torch.reshape(r, (r.shape[0], -1))
            r_m = torch.sub(r, m)
            r_m = r_m * r_m
            rn = torch.sum(r_m, dim=-1)
            varj += float(torch.mean(rn))
        varj /= k

    var_ /= float(n_shuffles)
    return np.sqrt(var_)


def shapes_permutations(filename, src_path, tar_path):
    x = loadh5(os.path.join(src_path, filename))
    n_shapes = x.shape[0]
    idx = torch.randperm(n_shapes)
    idx = np.asarray(idx, dtype=np.int)

    h5_fout = h5py.File(os.path.join(tar_path, filename), "w")
    h5_fout.create_dataset(
        'data', data=idx,
        compression='gzip', compression_opts=1,
        dtype='uint8')
    h5_fout.close()

def rot_permutations(tar_path, num_rots):
    filename = "rotations_permutations.h5"
    idx = torch.randperm(num_rots)
    idx = np.asarray(idx, dtype=np.int)

    h5_fout = h5py.File(os.path.join(tar_path, filename), "w")
    h5_fout.create_dataset(
        'data', data=idx,
        compression='gzip', compression_opts=1,
        dtype='uint8')
    h5_fout.close()


if __name__=="__main__":


    AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]



    # AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "cellphone.h5", "watercraft.h5"]

    # AtlasNetClasses = ["plane.h5"]
    AtlasNetShapesPath = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/valid"
    AtlasNetRotPath = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_valid"
    r_input_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations.h5"

    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full_multicategory"
    partial_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_partial"
    partial_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_partial_multicategory"

    """"
    for f in AtlasNetClasses:
        shapes_permutations(f, AtlasNetShapesPath, "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations")
    """
    # rot_permutations("I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024", 128)
    # exit(666)
    """
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_full_multicategory"


    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_consistency_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_consistency_full_multicategory"
    """
    # full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_consistency_full"
    # full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_consistency_full_multicategory"


    # ull_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full"
    # full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full_multicategory"


    # multicategory full shapes

    print("multi category")
    ma = 0.
    mb = 0.
    k = 0.
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], AtlasNetRotPath, full_multi_pred_path, batch_size=32)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], AtlasNetRotPath, full_multi_pred_path, batch_size=32)
        print("equivariance: ", b)
        mb += b
        k += 1.

    print("mean class consistency: ", ma / k)
    print("mean class equivariance: ", mb / k)


    print("category specific")
    ma = 0.
    mb = 0.
    k = 0.
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], AtlasNetRotPath, full_pred_path, batch_size=32)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], AtlasNetRotPath, full_pred_path, batch_size=32)
        print("equivariance: ", b)
        mb += b
        k += 1.

    print("mean class consistency: ", ma / k)
    print("mean class equivariance: ", mb / k)


    '''
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/spherical_cnns_full_multicategory"

    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/pca_full_multicategory"


    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/caca_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/caca_full_multicategory"

    """
    full_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full"
    full_multi_pred_path = "I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/preds/tfn_full_multicategory"
    """

    AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]

    print("multi category")
    ma = 0.
    mb = 0.
    mc = 0.
    k = 0.
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_multi_pred_path,
                                shapes_idx_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations", batch_size=32)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_multi_pred_path,
                                idx_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_permutations.h5", batch_size=32)
        print("equivariance: ", b)
        mb += b
        c = class_consistency_umetric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_multi_pred_path,
                                    idx_shapes_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations",
                                    idx_rots_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_permutations.h5",
                                    batch_size=32)
        mc += c
        print("u_consistency: ", c)

        k += 1.

    print("mean multi class consistency: ", ma / k)
    print("mean multi class equivariance: ", mb / k)
    print("mean multi class uconsistency: ", mc / k)


    AtlasNetClasses = ["plane.h5", "chair.h5"]

    AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5", "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]


    print("category specific")
    ma = 0.
    mb = 0.
    mc = 0.
    k = 0.
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_pred_path,
                                    shapes_idx_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations",
                                    batch_size=32)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_pred_path,
                                idx_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_permutations.h5",
                                batch_size=32)
        print("equivariance: ", b)
        mb += b
        c = class_consistency_umetric(AtlasNetClasses[i], AtlasNetShapesPath, r_input_path, full_pred_path,
                                    idx_shapes_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/shapes_permutations",
                                    idx_rots_path="I:/Datasets/Shapes/ShapeNetAtlasNetH5_1024/rotations_permutations.h5",
                                    batch_size=32)
        mc += c
        print("u_consistency: ", c)
        k += 1.

    print("mean class consistency: ", ma / k)
    print("mean class equivariance: ", mb / k)
    print("mean multi class uconsistency: ", mc / k)
    '''
