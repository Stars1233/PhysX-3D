import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d

import trimesh
import json
import torch
import kaolin
import math
from tools import VaryPoint
from collections import Counter



def _voxelize(file, sha256, output_dir):
    
    with open(os.path.join(output_dir, 'renders', sha256, 'transforms.json'),'r') as f:
        data=json.load(f)

    orimesh=trimesh.load(os.path.join('./phy_dataset',sha256[:-1],'model_tex.obj'))
    proind=np.load(os.path.join('./phy_dataset',sha256[:-1],'clip_ind_new.npy'))
    property_1=np.load(os.path.join('./phy_dataset',sha256[:-1],'clip.npy'))[np.int32(proind)]
    property_2=np.load(os.path.join('./phy_dataset',sha256[:-1],'otherproperty.npy'))[np.int32(proind)]

    
    orimesh.apply_transform(trimesh.transformations.scale_matrix(data['scale']))
    orimesh.apply_translation([data['offset'][0],data['offset'][2],-data['offset'][1]])
    
    
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    orimesh.apply_transform(rotation_matrix)
    orimesh.export(os.path.join(output_dir, 'renders', sha256, 'mesh_new.ply'))
    if os.path.exists(os.path.join(output_dir, 'renders', sha256, 'mesh.ply')):
        os.rename(os.path.join(output_dir, 'renders', sha256, 'mesh.ply'), os.path.join(output_dir, 'renders', sha256, 'mesh_old.ply'))

    mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, 'mesh_new.ply'))

    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices) 

    point_cloud = mesh.sample_points_poisson_disk(number_of_points=81920,init_factor=3,pcl=None)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5

    newver=torch.Tensor(vertices)
    oriver=torch.Tensor(np.array(orimesh.vertices))
    
    xyz,index_faces=kaolin.ops.mesh.sample_points(oriver.unsqueeze(0), torch.Tensor(orimesh.faces).long(), 102400)
    
    new=torch.Tensor(xyz[0])
    label=property_2[orimesh.faces][:,:,1]
    counts = (label[:, :, np.newaxis] == label[:, np.newaxis, :]).sum(axis=-1)
    result = counts.argmax(axis=1)

    
    value,ind=torch.sqrt(((newver[:,None,:].repeat(1,len(new),1)-new[None,:,:].repeat(len(newver),1,1))**2).sum(-1)).sort(axis=1)  
    
    newproperty_1=property_1[orimesh.faces][np.arange(len(result)),result,:,:][index_faces[0,ind[:,0]]]
    newproperty_2=property_2[orimesh.faces][np.arange(len(result)),result,:][index_faces[0,ind[:,0]]]

    

    np.save(os.path.join(output_dir, 'property', f'{sha256}1.npy'),newproperty_1)
    np.save(os.path.join(output_dir, 'property', f'{sha256}2.npy'),newproperty_2)

    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), vertices)
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'voxels'), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'property'), exist_ok=True)

    

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))

    
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == True]
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]
    
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')):

            pts = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
            records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')
    
    # process objects
    func = partial(_voxelize, output_dir=opt.output_dir)
    voxelized = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Voxelizing')
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)