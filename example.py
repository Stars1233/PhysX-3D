import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
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
import trimesh
def create_3d_arrow_mesh(position=[0, 0, 0], direction=[0, 0, 1], 
                         arrow_length=1.0, color=[255, 0, 0],
                         stem_radius=0.02, head_radius=0.05, head_length_ratio=0.07):
    """
    Creates a 3D arrow consisting of a cylinder (shaft) and a cone (head).
    :param position: Arrow base coordinates (default is the origin)
    :param direction: Arrow direction vector (normalized)
    :param arrow_length: Total arrow length
    :param color: Arrow color (RGB)
    :param stem_radius: Shaft (cylinder) radius
    :param head_radius: Head (cone) base radius
    :param head_length_ratio: Ratio of head length to total length (default is 0.2, i.e. 20%)
    :return: A trimesh.Trimesh object
    """

    direction = np.array(direction, dtype=np.float64)
    if np.allclose(direction, 0):
        raise ValueError("The direction vector cannot be the zero vector!")
    

    norm = np.linalg.norm(direction)
    direction = direction / norm

    head_length = arrow_length * head_length_ratio
    stem_length = arrow_length - head_length


    stem = trimesh.creation.cylinder(
        radius=stem_radius,
        height=stem_length,
        sections=32
    )


    head = trimesh.creation.cone(
        radius=head_radius,
        height=head_length,
        sections=32
    )


    head.apply_translation([0, 0, stem_length-1.55])


    arrow = trimesh.util.concatenate([stem, head])


    z_axis = np.array([0, 0, 1], dtype=np.float64)
    dot_product = np.dot(z_axis, direction)
    dot_product = np.clip(dot_product, -1.0, 1.0)  
    angle = np.arccos(dot_product)
    

    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-10:

        if dot_product < 0:

            rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:

            rotation = np.eye(4)
    else:
        axis = axis / axis_norm
        rotation = trimesh.transformations.rotation_matrix(angle, axis)

    arrow.apply_transform(rotation)
    
    arrow.apply_translation(position)
    arrow.visual.vertex_colors = color
    
    return arrow
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def draw_heatmap(data,max=1,min=0.0):
# heatmap setting
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--condpath", type=str, default='./example/table.png')
parser.add_argument("--savepath", type=str, default='./outputs_vis')
parser.add_argument("--question", type=str, default='Cushioned seat surface')
parser.add_argument("--question_type", type=int, default=0, help='0=basic discriptions, 1=function discriptions, 2=movement discriptions')
args = parser.parse_args()

modelpath='pretrain/diffusion'
# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained(modelpath)
pipeline.cuda()
savepath=os.path.join(args.savepath,modelpath)


clipmodel, preprocess = clip.load("ViT-L/14", jit=False)
clipmodel=clipmodel.eval().cuda()
os.makedirs(savepath, exist_ok=True)
image = Image.open(os.path.join(args.condpath))



# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 50,
    #     "cfg_strength": 3,
    # },
)
    

tokens=clip.tokenize(args.question).cuda()
info_emb=clipmodel.encode_text(tokens).float()

lang,phy=pipeline.models['slat_decoder_output'](outputs['mesh'][0].phy_property[:,16:,None,None],outputs['mesh'][0].phy_property[:,-16:,None,None])
phy=phy.squeeze()
# convert normalized data into physical data
phy[...,0]=(phy[...,0])*48+95
phy[...,2]=(phy[...,2])*2.8+2.3
phy[...,4]=(phy[...,4])*0.732-0.812
phy[...,-1]=(phy[...,-1])*4
target_lang=lang.reshape(-1,4,768)[:,args.question_type]
score=F.cosine_similarity(target_lang, info_emb, dim=1)
score=(score-score.min())/(score.max()-score.min())

material_min,material_max=phy[...,2].min(),phy[...,2].max()
material=(phy[...,2]-material_min)/(material_max-material_min)
affordance=(phy[...,1]-phy[...,1].min())/(phy[...,1].max()-phy[...,1].min())

num_group=round(float(phy[...,3].max()))+1



kinematic_parameters=[]
kinematic_type=[]
if num_group==1:
    outputs['mesh'][0].render_vis=torch.stack([affordance,material,score]).permute(1,0)
    print('fixed object')
else:
    kinematic_map=torch.zeros(len(material),(num_group-1)*2).cuda().long()
    for group_ind in range(1,num_group):
        kinematic_map[:,(group_ind-1)*2]=(phy[...,3]>group_ind-0.5)&(phy[...,3]<group_ind+0.5)
        kinematic_map[:,(group_ind-1)*2+1]=(phy[...,3]>(phy[...,4][kinematic_map[:,(group_ind-1)*2]]-0.5))&(phy[...,3]<(phy[...,4][kinematic_map[:,(group_ind-1)*2]]+0.5))
        if kinematic_map[:,(group_ind-1)*2+1].sum()<100:
            kinematic_map[:,(group_ind-1)*2+1]=phy[...,3]<-0.5

        kinematic_parameters.append(phy[...,5:-1][kinematic_map[(group_ind-1)*2]].mean(0))
        kinematic_type.append(phy[...,-1][kinematic_map[(group_ind-1)*2]].mean(0))

    outputs['mesh'][0].render_vis=torch.stack([affordance,material,score]).permute(1,0)
    outputs['mesh'][0].render_vis=torch.cat([outputs['mesh'][0].render_vis,kinematic_map],1)

video = render_utils.render_video(outputs['mesh'][0],num_frames=30)


video1=[]
video2=[]
video3=[]
kine1=[]
kine2=[]
for i in range(len(video['rendervis'])):
    vis=video['rendervis'][i]*video['mask'][i]
    img_0=draw_heatmap((vis[0]).detach().cpu().numpy())
    img_1=draw_heatmap(vis[1].detach().cpu().numpy(),float(material_max),float(material_min))
    img_2=draw_heatmap(vis[2].detach().cpu().numpy())
    if len(vis)>3:
        img_3=draw_heatmap(vis[3].detach().cpu().numpy(),1,0)
        img_4=draw_heatmap(vis[4].detach().cpu().numpy(),1,0)
        kine1.append(img_3)
        kine2.append(img_4)

    video1.append(img_0)
    video2.append(img_1)
    video3.append(img_2)
    
    

print('Physical scale: ',phy[...,0].mean().detach().cpu().numpy())
imageio.mimsave(os.path.join(savepath,"rgb.mp4"), video['color'], fps=30)
imageio.mimsave(os.path.join(savepath,"affordance.mp4"), video1, fps=30)
imageio.mimsave(os.path.join(savepath,"material.mp4"), video2, fps=30)
imageio.mimsave(os.path.join(savepath,"description.mp4"), video3, fps=30)
if len(kine1)>0:
    imageio.mimsave(os.path.join(savepath,"kinematic_child.mp4"), kine1, fps=30)
if len(kine2)>0:    
    imageio.mimsave(os.path.join(savepath,"kinematic_parent.mp4"), kine2, fps=30)



glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(os.path.join(savepath,"texture.glb"))

if len(kinematic_parameters)>0:
    mesh=trimesh.load(os.path.join(savepath,"texture.glb"),force='mesh')
    kinematic=kinematic_parameters[0].detach().cpu().numpy()
    arrow_mesh=create_3d_arrow_mesh(position=kinematic[3:6],direction=kinematic[:3],arrow_length=3,color=[255,0,0],stem_radius=0.01,head_radius=0.035)
    combined = trimesh.util.concatenate([mesh,arrow_mesh])
    combined.export(os.path.join(savepath,"kinematic.obj"))
    print('kinematic range: ',kinematic[-2],kinematic[-1])
    kinematics=float(kinematic_type[0].detach().cpu().numpy())
    if round(kinematics)==1:
        print('kinematic type: ','D. Hinge joint')
    elif round(kinematics)==2:
        print('kinematic type: ','C. Revolute joints')
    elif round(kinematics)==3:
        print('kinematic type: ','B. Prismatic joints ')
    elif round(kinematics)==4:
        print('kinematic type: ','A. No movement constraints')
    else:
        print('kinematic type: ','E. Rigid joint')


