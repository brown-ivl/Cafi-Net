from pkg_resources import declare_namespace
import torch
import os
from models import *
from utils import *
from utils.train_utils import random_rotate, mean_center, perform_rotation, orthonormalize_basis
from utils.losses import equilibrium_loss , localization_loss_new
import datasets
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import hydra
from utils.pointcloud_utils import kdtree_indexing, save_pointcloud, convert_yzx_to_xyz_basis, get_equivariant_density, rotate_density, sq_distance_mat, save_density, get_gradient_density,create_mask
from scipy.spatial.transform import Rotation
from random import randint
import h5py
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import open3d as o3d
from pytorch3d.loss import chamfer_distance
import random, sys

class Canonical_fields_trainer(pl.LightningModule):
    '''
    Segmentation trainer to mimic NeSF
    '''

    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters()
        self.hparam_config = configs
        self.Cafinet = getattr(eval(self.hparam_config.model["file"]), 
                                  self.hparam_config.model["type"])(**self.hparam_config.model["args"])
        
        self.hparam_config.dataset.args.dataset_path = os.path.join(hydra.utils.get_original_cwd(), self.hparam_config.dataset.args.dataset_path)
        self.loss_weights = self.hparam_config.loss


    
    def train_dataloader(self):
        train_data_set = getattr(getattr(datasets, self.hparam_config.dataset.file), self.hparam_config.dataset.type)(**self.hparam_config.dataset.args)
        train_dataloader = DataLoader(train_data_set, **self.hparam_config.dataset.loader.args, shuffle = True)

        return train_dataloader

    def val_dataloader(self):
        val_data_set = getattr(getattr(datasets, self.hparam_config.val_dataset.file), self.hparam_config.val_dataset.type)(**self.hparam_config.val_dataset.args)
        val_dataloader = DataLoader(val_data_set, **self.hparam_config.val_dataset.loader.args, shuffle = False)
        return val_dataloader


    def forward_pass(self, batch, batch_idx, return_outputs=False, rot_mat=None):
        

        density = batch["density"][0].clone()
        coords  = batch["coords"][0].clone()
        
        

        
        iter_ =2   
        if return_outputs == True :
            iter_ = 1
        
        loss_dictionary_list = []   
        frame_list = []
        inv_embed_list = []
        for ii in range(iter_):
        
            if rot_mat is None :
                rotation_1 = torch.from_numpy(Rotation.random(density.shape[0]).as_matrix()).type_as(density)
            else:
                rotation_1 = rot_mat.type_as(density)
                
            density = batch["density"][ii].clone()
            coords  = batch["coords"][ii].clone()
            density = rotate_density(rotation_1, density)
            
            B,_,_,_ = density.shape
            shape_ = density.shape
             
            mask_density = create_mask(density).cuda()   
            
            out_dict = self.Cafinet(coords, density)
            
            batch["eq_input_1"] = coords
            batch["x_input"] = density
            batch["density_rotated_1"] = density        
            batch["rotation_1"] = rotation_1
            batch["mask_density"] = mask_density


            B,_,_,_,_ = out_dict["coords"].shape
            
            if return_outputs:
                out_dict["x_input"] = density
                loss_dictionary, frame = self.compute_loss(batch, out_dict, return_frame = True)

                out_dict["E"] = frame
                out_dict["coords"] = out_dict["coords"].reshape(B,-1,3)
                out_dict["input_rotation"] = rotation_1
                return loss_dictionary, out_dict
            else:
                loss_dictionary , frame  = self.compute_loss(batch, out_dict, return_frame=True)
                loss_dictionary_list.append(loss_dictionary)
                frame_list.append(frame)
                pts_list = []
                for _indx in range(B):
                    pts_list.append(out_dict["coords"][_indx][mask_density[_indx] >= 0.5].unsqueeze(0))
                inv_embed_list.append(pts_list)
                        
        
        for ii in range(1,len(loss_dictionary_list)):
            for key in loss_dictionary_list[0].keys():
                loss_dictionary_list[0][key] = loss_dictionary_list[0][key] + loss_dictionary_list[ii][key]
        
        for key in loss_dictionary_list[0].keys():
                loss_dictionary_list[0][key] = loss_dictionary_list[0][key] / (len(loss_dictionary_list))
        
        inv_loss = 0.0
        for _indx in range(B):
            inv_loss = inv_loss + chamfer_distance(inv_embed_list[0][_indx],inv_embed_list[1][_indx])[0]
            
        inv_loss = inv_loss / float(B)
        
       
        loss_dictionary_list[0]["loss"]  = loss_dictionary_list[0]["loss"] + inv_loss
        loss_dictionary_list[0]["inv_loss"]  = inv_loss
        
        return loss_dictionary_list[0]

    def compute_loss(self, batch, outputs, return_frame = False):
        """
        Computing losses for 
        """
        
        loss_dictionary = {}
        loss = 0.0
        
        out_density_1 = batch["mask_density"]
        rotation_1 = batch["rotation_1"]


        # First branch
        density_canonical = outputs["points_inv"]
        basis_1 = outputs["E"]
        canonical_coords = outputs["coords"]
        basis_1 = torch.stack(basis_1, dim = 1) # B, num_frames, 3, 3
       
        B, H, W, D, _3 = canonical_coords.shape

        orth_basis_1 = (orthonormalize_basis(basis_1))
        basis_canonical_to_input = convert_yzx_to_xyz_basis((orth_basis_1))
        eq_input_1 =batch["eq_input_1"].reshape(B,-1,3)
        

        
    
       

        B, num_frames = orth_basis_1.shape[0], orth_basis_1.shape[1]
       
        eq_coords_pred = torch.einsum("bfij, bhwdj->bfhwdi", basis_canonical_to_input, canonical_coords).reshape(B, num_frames, -1, _3)

        
        mask_object = 1.0*(out_density_1 >= 0.5).reshape(B, 1, -1)

        
        
        error_full = torch.sum(torch.sqrt(torch.sum(torch.square(eq_coords_pred - eq_input_1[:, None]), -1) + 1e-8) * mask_object, -1) / (torch.sum(mask_object, -1) + 1e-10)
                    


        values, indices = torch.topk(-error_full, k = 1)
        
        
        orth_basis_frames = basis_canonical_to_input
        basis_canonical_to_input = basis_canonical_to_input[torch.arange(indices.shape[0]), indices[:, 0]]
        

        eq_input_pred_best = torch.einsum("bij, bhwdj->bhwdi", basis_canonical_to_input, canonical_coords).reshape(B, -1, _3)
        l2_loss = torch.mean(torch.sum(torch.sqrt(torch.sum(torch.square(eq_input_pred_best - eq_input_1), -1) + 1e-8) * mask_object.squeeze(1),-1) / (torch.sum(mask_object.squeeze(1),-1) + 1e-10))
        
        
        
        chamfer_loss = 0.0
        for idx in range(B):
            int_mask = mask_object[idx].permute(1,0)
            int_pts_idx,_ = torch.where(int_mask >=0.5)
            chamfer_loss = chamfer_loss + chamfer_distance(eq_input_pred_best[idx][int_pts_idx,:].unsqueeze(0),eq_input_1[idx][int_pts_idx,:].unsqueeze(0))[0]
            
        chamfer_loss = chamfer_loss / float(B)  
      
       
        orth_loss = torch.mean(torch.abs(basis_1 - orth_basis_1.detach()))
        I = torch.eye(num_frames).type_as(basis_1).unsqueeze(0)
        ones = torch.ones(B, num_frames, num_frames).type_as(basis_1)
        weights = ones - I
        separation_loss = torch.sum(torch.mean(torch.mean(torch.exp(-(torch.abs(basis_1[:, :, None] - basis_1[:, None]))), -1), -1) * weights) / (torch.sum(weights) + 1e-10)
        
        
       

        if self.loss_weights.l2_loss > 0.0:
            loss += self.loss_weights.l2_loss * l2_loss
        
        if self.loss_weights.chamfer_loss > 0.0:
            loss += self.loss_weights.chamfer_loss * chamfer_loss
        

        if self.loss_weights.separation_loss > 0.0:
            loss += self.loss_weights.separation_loss * separation_loss


        if self.loss_weights.orth_loss > 0.0:
            loss += self.loss_weights.orth_loss * orth_loss

      
        
        loss_dictionary["loss"] = loss
        loss_dictionary["l2_loss"] = l2_loss
        loss_dictionary["orth_loss"] = orth_loss  
        loss_dictionary["separation_loss"] = separation_loss
        loss_dictionary["chamfer_loss"] = chamfer_loss  
        
        if return_frame:
            return loss_dictionary, basis_canonical_to_input
        return loss_dictionary

    def training_step(self, batch, batch_idx):

        loss_dictionary = self.forward_pass(batch, batch_idx)
        self.log_loss_dict(loss_dictionary)

        return loss_dictionary["loss"]

    def validation_step(self, batch, batch_idx):

        loss_dictionary = self.forward_pass(batch, batch_idx)
        self.log_loss_dict(loss_dictionary, val = True)

        return loss_dictionary["loss"]


    def configure_optimizers(self):

        optimizer1 = getattr(torch.optim, self.hparam_config.optimizer.type)(list(self.Cafinet.parameters()), **self.hparam_config.optimizer.args)
        scheduler1 = getattr(torch.optim.lr_scheduler, self.hparam_config.scheduler.type)(optimizer1, **self.hparam_config.scheduler.args)

        return [optimizer1], [scheduler1]



    def log_loss_dict(self, loss_dictionary, val = False):

        for key in loss_dictionary:
            if val:
                self.log("val_" + key, loss_dictionary[key], **self.hparam_config.logging.args)
            else:
                self.log(key, loss_dictionary[key], **self.hparam_config.logging.args)


    def test_step(self, x):
        '''
        Input:
            x - B, N, 3
        Output:
            output_dictionary - dictionary with all outputs and inputs
        '''
        output_dictionary = self.forward_pass(x)

        return output_dictionary


    def save_outputs(self, save_file_name, out_dict):
        """
        Save outputs to file
        """
        save_keys = ["x_input", "canonical_density"]
        for key in out_dict:
            if key in save_keys:
                pcd_name = save_file_name + "_" + key + ".ply"
                save_density(out_dict[key], pcd_name)
        
    
    def loadh5(self,path):
        fx_input = h5py.File(path, 'r')
        x = fx_input['data'][:]
        fx_input.close()
        return x

        
    def save_h5(self,h5_filename, data,data_dtype='float32'):
        h5_fout = h5py.File(h5_filename,"w")

        h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
        h5_fout.close()
    
    def run_test(self, cfg ,dataset_num = 1, save_directory = "./pointclouds", max_iters = None, skip = 1 , num_rots=1):
        
        self.hparam_config = cfg

        self.hparam_config.val_dataset.loader.args.batch_size = 1
        loader = self.val_dataloader()

        if max_iters is not None:
            max_iters = min(max_iters, len(loader))
        else:
            max_iters = len(loader)
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i % skip == 0:
                    batch["density"][0] = batch["density"][0].cuda()
                    batch["coords"][0]  = batch["coords"][0].cuda()
                    
                    x = batch
                    for _ in range(num_rots):
                        random_rots = torch.from_numpy(Rotation.random(num_rots).as_matrix())
                        rot_mat = random_rots[_].reshape(1,3,3)                      
                        out, output_dict = self.forward_pass(x, 0, return_outputs = True, rot_mat=torch.inverse(rot_mat))
                        density_1 = output_dict["x_input"]
                        basis_1 = output_dict["E"]
                        canonical_density = rotate_density(basis_1, density_1)
                        output_dict["canonical_density"] = canonical_density                      
                        save_file_name = os.path.join(save_directory, "") + str(_) + "_ "+"%d" % i 
                        self.save_outputs(save_file_name, output_dict)

              
    def run_metrics(self, cfg ,dataset_num = 1, save_directory = "./pointclouds", max_iters = None, skip = 1):

        self.hparam_config = cfg

        self.hparam_config.val_dataset.loader.args.batch_size = 1
        loader = self.val_dataloader()

        if max_iters is not None:
            max_iters = min(max_iters, len(loader))
        else:
            max_iters = len(loader)
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        random_rots = self.loadh5(self.hparam_config.test_args.rotation_file)
        random_rots = torch.from_numpy(random_rots)
        
        category_name = self.hparam_config.test_args.category_name
        h5_filename = save_directory + "/"+ category_name +"_rotations.h5"
        self.save_h5(h5_filename, random_rots)
        
        

        canonical_frame=[]
        with torch.no_grad():
            for i, batch in enumerate(loader):
                print("processing for Nerf Model ",i)
                if i % skip == 0:
                    batch["density"][0] = batch["density"][0].cuda()
                    batch["coords"][0] = batch["coords"][0].cuda()
                    x = batch
                    canonical_frame_obj = []
                    
                    for _ in range(random_rots.shape[0]):
                        rot_mat = random_rots[_].reshape(1,3,3)                                            
                        out, output_dict = self.forward_pass(x, 0, return_outputs = True, rot_mat=torch.inverse(rot_mat))
                        basis_1 = output_dict["E"]
                        canonical_frame_obj.append(torch.inverse(basis_1.squeeze(0)))
                    
                    canonical_frame.append(torch.stack(canonical_frame_obj))
            

            canonical_frame = torch.stack(canonical_frame)
            h5_filename = save_directory + "/"+ category_name +"_canonical.h5"
            self.save_h5(h5_filename, canonical_frame.cpu().detach().numpy())
    
    
    def run_canonica_render(self, cfg ,dataset_num = 1, save_directory = "./pointclouds", max_iters = None, skip = 1):
        
        self.hparam_config = cfg

        self.hparam_config.val_dataset.loader.args.batch_size = 1
        loader = self.val_dataloader()

        if max_iters is not None:
            max_iters = min(max_iters, len(loader))
        else:
            max_iters = len(loader)
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        
        category_name = self.hparam_config.test_args.category_name
        
        rotation_list = []
        canonical_frame=[]
        file_name_list=[]
        with torch.no_grad():
            for i, batch in enumerate(loader):
                print("processing for Nerf Model ",i)
                if i % skip == 0:
                    batch["density"][0] = batch["density"][0].cuda()
                    batch["coords"][0] = batch["coords"][0].cuda()
                    file_name_list.append(batch["file_path"][0])
                    x = batch
                    canonical_frame_obj = []
                    for _ in range(1):
                        rot_mat = torch.eye(3).reshape(1,3,3).to(torch.float32).cuda() 
                        random_rots = torch.from_numpy(Rotation.random(1).as_matrix())
                        rot_mat = random_rots[_].reshape(1,3,3) 
                        rotation_list.append(rot_mat)
                        out, output_dict = self.forward_pass(x, 0, return_outputs = True, rot_mat=torch.inverse(rot_mat))
                        basis_1 = output_dict["E"]
                        canonical_frame_obj.append(torch.inverse(basis_1.squeeze(0)))
                    
                    canonical_frame.append(torch.stack(canonical_frame_obj))
            
            
            
            rotation_list = torch.stack(rotation_list,axis=0)
            h5_filename = save_directory + "/"+ category_name +"_input_rot.h5"
            self.save_h5(h5_filename, rotation_list.cpu().detach().numpy())
            
            file_name = save_directory + "/" + category_name +"_files.txt"
            file_prt = open(file_name,"w+")
            for _ in file_name_list:
                file_prt.write(_+"\n")
            file_prt.close()

            
            canonical_frame = torch.stack(canonical_frame)
            h5_filename = save_directory + "/"+ category_name +"_canonical.h5"
            self.save_h5(h5_filename, canonical_frame.cpu().detach().numpy())
            
