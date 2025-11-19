import os
import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import utils3d.torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.colors import ListedColormap
import cv2
from scipy.ndimage import generic_filter
import logging
import ipdb
import trimesh
import json
from trellis.representations.mesh.cube2mesh import MeshExtractResult

def rotate_vertices_about_axis(verts: np.ndarray,
                               axis: np.ndarray,
                               angle: float,
                               origin: np.ndarray | None = None,
                               degrees: bool = False) -> np.ndarray:

    v = np.asarray(verts, dtype=float)
    a = np.asarray(axis, dtype=float).reshape(3)
    if origin is None:
        o = np.zeros(3, dtype=float)
    else:
        o = np.asarray(origin, dtype=float).reshape(3)

    norm = np.linalg.norm(a)
    if norm < 1e-12:
        raise ValueError("axis is zero vector")
    k = a / norm

    theta = np.deg2rad(angle) if degrees else angle
    ct = np.cos(theta)
    st = np.sin(theta)
    kx, ky, kz = k

    K = np.array([[    0, -kz,  ky],
                  [  kz,   0, -kx],
                  [ -ky,  kx,   0]], dtype=float)
    kkT = np.outer(k, k)
    R = ct*np.eye(3) + (1-ct)*kkT + st*K

    vp = v - o
    vr = vp @ R.T
    return vr + o





def draw_heatmap(data,max=1,min=0.0):
    fig = plt.figure(figsize=(7, 5))  
    ax = fig.add_subplot(111)
    cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  
    jet = cm.get_cmap('jet', 256)
    jet_colors = jet(np.linspace(0, 1, 256))
    jet_colors[0] = [0, 0, 0, 1]  
    jet_black_bg = ListedColormap(jet_colors)

    # initialize heatmap
    im = ax.imshow(np.random.rand(512,512), cmap=jet_black_bg, vmin=0, vmax=1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([str(min), str(0.5*(max-min)), str(1.0*(max))])
    ax = fig.add_axes([0.08, 0.15, 0.8, 0.7]) 
    ax.axis('off')

    im.set_data(data)
    fig.canvas.draw()  
    img_array = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
    return img_array




jsonpath='./PhysXNet/finaljson'
meshpath='./PhysXNet/partseg'

namelist=np.load('./testset.npy') #testlist

for name in namelist:
    with open(os.path.join(jsonpath,name+'.json'),'r') as fp:
        jsondata=json.load(fp)
    question=jsondata['parts'][0]['Basic_description']

    savepath=os.path.join('./outputs_vis_gt',name)

    os.makedirs(savepath, exist_ok=True)

    resultsavepath4=os.path.join(savepath,'mesh')
    os.makedirs(resultsavepath4, exist_ok=True)
    index=0

    if len(jsondata['group_info'])>1:

        childlist=jsondata['group_info']['1'][0]
        
        if jsondata['group_info']['1'][1]=='0':
            parentlist=jsondata['group_info'][jsondata['group_info']['1'][1]]
        else:
            parentlist=jsondata['group_info'][jsondata['group_info']['1'][1]][0]
        

        childrenobj=trimesh.Trimesh([])
        for meshname in childlist:
            eachpart1=trimesh.load(os.path.join(meshpath,name,'objs',str(meshname)+'.obj'))
            childrenobj = trimesh.util.concatenate([childrenobj,eachpart1])

        parentrenobj=trimesh.Trimesh([])
        for meshname in parentlist:
            eachpart1=trimesh.load(os.path.join(meshpath,name,'objs',str(meshname)+'.obj'))
            parentrenobj = trimesh.util.concatenate([parentrenobj,eachpart1])

        kinematic=jsondata['group_info']['1'][-2]
        position=kinematic[3:6]
        direction=kinematic[:3]
        if jsondata['group_info']['1'][-1]=='C':
            for location in np.linspace(kinematic[-2],kinematic[-1],4):
                new=rotate_vertices_about_axis(childrenobj.vertices,np.array(direction),location*180,np.array(position), degrees=True)

                newchildmesh=trimesh.Trimesh(vertices=new,faces=childrenobj.faces.copy())
                combined = trimesh.util.concatenate([newchildmesh,parentrenobj])
                combined.export(os.path.join(resultsavepath4,str(index)+'.obj'))
                index+=1
        elif jsondata['group_info']['1'][-1]=='B':
            for location in np.linspace(kinematic[-2],kinematic[-1],4):

                    newchildmesh=trimesh.Trimesh(vertices=childrenobj.vertices+np.array(direction)*location,faces=childrenobj.faces.copy())
                    combined = trimesh.util.concatenate([newchildmesh,parentrenobj])
                    combined.export(os.path.join(resultsavepath4,str(index)+'.obj'))
                    index+=1

    str_list=jsondata['dimension'].split(' ')[0].split('*')
    sorted_list = sorted(str_list, key=float, reverse=True)
    scaling=float(sorted_list[0])

    np.save(os.path.join(savepath,'scale.npy'),np.array(scaling))

    allrenobj=trimesh.Trimesh([])
    allsourcedata=np.zeros((0,3)) #affordance,material,score

    
    
    for part in range(len(jsondata['parts'])):

        eachpart1=trimesh.load(os.path.join(meshpath,name,'objs',str(meshname)+'.obj'))
        allrenobj = trimesh.util.concatenate([eachpart1,allrenobj])

        sourcedata=np.zeros((len(eachpart1.vertices),3))
        sourcedata[:,0]=jsondata['parts'][part]['priority_rank']
        sourcedata[:,1]=float(jsondata['parts'][part]['density'].split(' ')[0])

        description_ind=np.random.randint(len(jsondata['parts']))
        
        np.save(os.path.join(savepath,'des_index.npy'),jsondata['parts'][part]['Basic_description'])

        if part==description_ind:
            sourcedata[:,2]=1
        else:
            sourcedata[:,2]=0
        allsourcedata=np.concatenate([sourcedata,allsourcedata])


    output=MeshExtractResult(
        torch.Tensor(allrenobj.vertices).cuda(),
        torch.Tensor(allrenobj.faces).cuda(),
        vertex_attrs=None,
        res=64,
        phy_property=None,
        render_vis=torch.Tensor(allsourcedata).cuda()
    )

    video = render_utils.render_video_gt(output,num_frames=30)


    video1=[]
    video2=[]
    video3=[]
    kine1=[]
    kine2=[]

    resultsavepath1=os.path.join(savepath,'affordance')
    resultsavepath2=os.path.join(savepath,'material')
    resultsavepath3=os.path.join(savepath,'description')

    os.makedirs(resultsavepath1, exist_ok=True)
    os.makedirs(resultsavepath2, exist_ok=True)
    os.makedirs(resultsavepath3, exist_ok=True)
    for i in range(len(video['rendervis'])):
        vis=video['rendervis'][i]*video['mask'][i]
 
        img_0=((vis[0]).detach().cpu().numpy())
        img_1=(vis[1].detach().cpu().numpy())
        img_2=(vis[2].detach().cpu().numpy())
        

        np.save(os.path.join(resultsavepath1,str(i)+'.npy'),img_0)
        np.save(os.path.join(resultsavepath2,str(i)+'.npy'),img_1)
        np.save(os.path.join(resultsavepath3,str(i)+'.npy'),img_2)

    



