import open3d as o3d
import numpy as np
import os
from scipy.spatial import ConvexHull
import json
import torch
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
import trimesh

import logging
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



def fit_planes(pcd, distance_threshold=0.01, max_planes=5, min_inliers=100):
    """
    Use RANSAC hierarchical segmentation to fit multiple planes
    Parameters:
    pcd (open3d.geometry.PointCloud): input point cloud
    distance_threshold (float): RANSAC inlier distance threshold
    max_planes (int): maximum number of planes
    min_inliers (int): minimum inlier threshold for a plane
    Returns:
    planes (list): list of plane parameters, each plane is in the format of (A, B, C, D)
    """
    planes = []
    points=[]
    remaining_pcd = pcd
    
    for _ in range(max_planes):
        if len(remaining_pcd.points) < 3 * min_inliers:  # Stop when the remaining points are insufficient
            break
        
        # Split the current largest plane
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < min_inliers:
            break  
        
        planes.append(plane_model)
        points.append(inliers)

        remaining_pcd = remaining_pcd.select_down_sample(inliers, invert=True)
    
    return planes,points


def rotate_vector(v, k, theta):
    """
    Calculate the new vector after the vector v is rotated around the axis k by the angle theta
    :param v: vector to be rotated (1x3 numpy array)
    :param k: unit vector of the rotation axis (1x3 numpy array)
    :param theta: rotation angle (radians)
    :return: rotated vector (1x3 numpy array)
    """

    k = k / np.linalg.norm(k)

    v_rot = (v * np.cos(theta) +
             np.cross(k, v) * np.sin(theta) +
             k * np.dot(k, v) * (1 - np.cos(theta)))
    
    return v_rot




def create_3d_arrow_mesh(position=[0, 0, 0], direction=[0, 0, 1], 
                         arrow_length=1.0, color=[255, 0, 0],
                         stem_radius=0.02, head_radius=0.05, head_length_ratio=0.07):
    """
    Create a 3D arrow consisting of a cylinder (arrow shaft) and a cone (head).
    :param position: Arrow base coordinates (default origin)
    :param direction: Arrow direction vector (needs to be normalized)
    :param arrow_length: Arrow total length
    :param color: Arrow color (RGB)
    :param stem_radius: Arrow shaft (cylinder) radius
    :param head_radius: Head (cone) bottom radius
    :param head_length_ratio: The ratio of head length to total length (default 0.2, i.e. 20%)
    :return: trimesh.Trimesh object
    """
    # --- Check direction vector validity ---
    direction = np.array(direction, dtype=np.float64)
    if np.allclose(direction, 0):
        raise ValueError("The direction vector cannot be the zero vector!")
    
    # --- Normalized direction vector ---
    norm = np.linalg.norm(direction)
    direction = direction / norm

    # --- Calculate the actual length of the head and shaft ---
    head_length = arrow_length * head_length_ratio
    stem_length = arrow_length - head_length

    # --- Create the arrow shaft ---
    stem = trimesh.creation.cylinder(
        radius=stem_radius,
        height=stem_length,
        sections=32
    )

    # --- Create the head ---
    head = trimesh.creation.cone(
        radius=head_radius,
        height=head_length,
        sections=32
    )

    # --- Adjust head position---
    head.apply_translation([0, 0, stem_length-1.05])

    # --- Combine the shaft and head ---
    arrow = trimesh.util.concatenate([stem, head])

    # --- Calculate the rotation matrix ---
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    dot_product = np.dot(z_axis, direction)
    dot_product = np.clip(dot_product, -1.0, 1.0)  
    angle = np.arccos(dot_product)
    
    # Calculate the axis of rotation
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-10:
        # Direction parallel (or anti-parallel) to the Z axis
        if dot_product < 0:
            # Rotate 180 degrees around any vertical axis when antiparallel
            rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            # No rotation when parallel
            rotation = np.eye(4)
    else:
        axis = axis / axis_norm
        rotation = trimesh.transformations.rotation_matrix(angle, axis)
    
    # Rotation
    arrow.apply_transform(rotation)
    
    # --- Move the arrow to the target position ---
    arrow.apply_translation(position)
    
    # --- Set color ---
    arrow.visual.vertex_colors = color
    
    return arrow,rotation

def create_3d_arrow_mesh_rotation(position=[0, 0, 0], direction=[0, 0, 1], 
                         arrow_length=1.0, color=[255, 0, 0],
                         stem_radius=0.02, head_radius=0.05, head_length_ratio=0.07):
    """
    Create a 3D arrow consisting of a cylinder (arrow shaft) and a cone (head).
    :param position: Arrow base coordinates (default origin)
    :param direction: Arrow direction vector (needs to be normalized)
    :param arrow_length: Arrow total length
    :param color: Arrow color (RGB)
    :param stem_radius: Arrow shaft (cylinder) radius
    :param head_radius: Head (cone) bottom radius
    :param head_length_ratio: The ratio of head length to total length (default 0.2, i.e. 20%)
    :return: trimesh.Trimesh object
    """
    # --- 检查方向向量合法性 ---
    direction = np.array(direction, dtype=np.float64)
    if np.allclose(direction, 0):
        raise ValueError("The direction vector cannot be the zero vector!")
    
    # --- Normalized direction vector ---
    norm = np.linalg.norm(direction)
    direction = direction / norm

    # --- Calculate the actual length of the head and shaft ---
    head_length = arrow_length * head_length_ratio
    stem_length = arrow_length - head_length

    # --- Create the arrow shaft ---
    stem = trimesh.creation.cylinder(
        radius=stem_radius,
        height=stem_length,
        sections=32
    )

    # --- Creating the head ---
    head = trimesh.creation.cone(
        radius=head_radius,
        height=head_length,
        sections=32
    )

    # --- Adjust head position ---
    head.apply_translation([0, 0, stem_length-1.45])

    # --- Combine the shaft and head ---
    arrow = trimesh.util.concatenate([stem, head])

    # --- Calculate the rotation matrix ---
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    dot_product = np.dot(z_axis, direction)
    dot_product = np.clip(dot_product, -1.0, 1.0)  
    angle = np.arccos(dot_product)
    
    # Calculate the axis of rotation
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-10:
        # Direction parallel (or anti-parallel) to the Z axis
        if dot_product < 0:
            # Rotate 180 degrees around any vertical axis when antiparallel
            rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            # No rotation when parallel
            rotation = np.eye(4)
    else:
        axis = axis / axis_norm
        rotation = trimesh.transformations.rotation_matrix(angle, axis)
    
    # Rotation
    arrow.apply_transform(rotation)
    
    # --- Move the arrow to the target position ---
    arrow.apply_translation(position)
    
    # --- Set color ---
    arrow.visual.vertex_colors = color
    
    return arrow,rotation


def vector_angle_np(a, b):
    a = np.array(a)
    b = np.array(b)
    # Check Dimensions
    if a.shape != b.shape:
        raise ValueError("Vector dimensions are different")
    # Calculate dot product and modulus
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Handling zero vectors
    if norm_a == 0 or norm_b == 0:
        
        raise ValueError("There is a zero vector and the angle cannot be calculated")
    # Calculating Angles
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # limite the range
    degrees = np.degrees(np.arccos(cos_theta))
    return degrees


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--range", type=int, default=3000)
    args = parser.parse_args()

    partsegpath='./physxnet/partseg'
    jsonpath='./output_physxnet/1_gpt_annotation'
    targetfiles=os.listdir(jsonpath)
    targetfiles = sorted(targetfiles, key=lambda x: int(x.split('.')[0]))
    
    savepath='./output_physxnet'
    targetfiles=targetfiles[args.index*args.range:(args.index+1)*args.range]
    

    os.makedirs(savepath, exist_ok=True)
    os.makedirs(os.path.join(savepath,'objs_rotate'), exist_ok=True)
    os.makedirs(os.path.join(savepath,'objs_point'), exist_ok=True)
    os.makedirs(os.path.join(savepath,'objs_mov'), exist_ok=True)
    os.makedirs(os.path.join(savepath,'objs_axis'), exist_ok=True)
    
    color_library=np.array([[255,0,0],[70,70,70],[ 0,255,0],[ 0,0,255],[255, 165, 0],[75, 0, 130],[0, 255, 255],[255, 0, 255],\
    [117, 20, 11],[ 0, 100, 100],[146,101, 49],[254,249,204],[203, 65, 84],[100,128, 37],\
    [70,70,70],[215,191,249]])
    logger = get_logger(os.path.join(savepath,'exp_2vis'+str(args.index)+'.log'),verbosity=1)

    logger.info('start')


    rotation_arrow=trimesh.load('./arrow_norm.obj',force='mesh')
    
    

    for target in targetfiles:
            target=target[:-5]
            logger.info('begin :'+target)

            # --- Load json file ---
        
            jsonfile=os.path.join(jsonpath,target+'.json')

            with open(jsonfile,'r') as f:
                data=json.load(f)

            partnum=len(data['parts'])
            existpart=[]
            existpartpair=[]

            for eachpart in data['parts']:
                nei_parts=eachpart['neighbors']
                target_part=eachpart['label']
                num_mov=0
                for nei_part in nei_parts:

                    # --- Choose parts that need kinematic information (B,C,D) ---

                    if (nei_part['movement_type']=='B' or nei_part['movement_type']=='C' or nei_part['movement_type']=='D'):
                        if nei_part['child_label']==target_part:
                            
    
                            child=nei_part['child_label']
                            if os.path.exists(os.path.join(savepath,'objs_mov',target+'-'+str(child)+'.npy')) or os.path.exists(os.path.join(savepath,'objs_rotate',target+'-'+str(child)+'.npy')) or os.path.exists(os.path.join(savepath,'objs_point',target+'-'+str(child)+'.npy')):
                                logger.info('skip :'+target)
                                num_mov+=1
                                continue
                            
                            if num_mov>0:
                                logger.info('repeat movement:'+target)

                            # --- Generate candidate axes for B ---

                            if nei_part['movement_type']=='B' and nei_part['child_label']==target_part:
                                num_mov+=1
                                parent=nei_part['parent_label']
                                child=nei_part['child_label']
                                
                                
                                if str(child) not in existpart:
                                
                                    existpart.append(str(child))
                                    existpartpair.append(str(child)+'_'+str(parent))
                                    childfile=trimesh.load(os.path.join(partsegpath,target,'objs',str(child)+'.obj'),force='mesh')

                                    otherlist=os.listdir(os.path.join(partsegpath,target,'objs'))
                                    otherlist=list(set(otherlist)-set([str(child)+'.obj']))
                                    for other in range(len(otherlist)):
                                        if other==0:
                                            othermesh=trimesh.load(os.path.join(partsegpath,target,'objs',otherlist[other]),force='mesh')
                                        else:
                                            othermesh = trimesh.util.concatenate([othermesh,trimesh.load(os.path.join(partsegpath,target,'objs',otherlist[other]),force='mesh')])

                                    value,ind=((torch.Tensor(othermesh.vertices)[None].repeat(len(childfile.vertices),1,1)-torch.Tensor(childfile.vertices)[:,None].repeat(1,len(othermesh.vertices),1))**2).sum(-1).min(0)

                                    error=0
                                    threshold=1e-2
                                    jointregion=torch.Tensor(othermesh.vertices)[torch.where(value.sqrt()<threshold)]
                                    while len(jointregion)<=6:
                                        
                                        jointregion=torch.Tensor(othermesh.vertices)[torch.where(value.sqrt()<threshold)]
                                        threshold+=1e-2


                                    if len(jointregion)>3:
                                        pcd = o3d.geometry.PointCloud()
                                        pcd.points = o3d.utility.Vector3dVector((jointregion).numpy())

                                        planes,innerpoint = fit_planes(pcd, distance_threshold=0.05, max_planes=1, min_inliers=int(len(jointregion)*0.1))
                                        
                                        para1,para2,para3,para4=planes[0]
                                        plane_point=jointregion[innerpoint[0]].numpy().mean(0)
                                        inner=[para1,para2,para3]*plane_point
                                        para4=-inner.sum()
                                        
                                        
                                        orilist=np.array([[1,0,0],[0,1,0],[0,0,1]])
                                        view=[]
                                        angle=[]
                                        for i in range(3):
                                            
                                            if vector_angle_np(planes[0][:3],orilist[i])>90:
                                                angle.append(180-vector_angle_np(planes[0][:3],orilist[i]))
                                                view.append(-planes[0][:3])
                                            else:
                                                angle.append(vector_angle_np(planes[0][:3],orilist[i]))
                                                view.append(planes[0][:3])
                                        planeview=view[np.array(angle).argmin()]    


                                        newplane_fine=[np.array([planeview[0],planeview[1],planeview[2],para4])]

                                        initview=np.cross(planeview,orilist[list(set([0,1,2])-set([np.array(angle).argmin()]))[0]])
                                        if vector_angle_np(initview,orilist[list(set([0,1,2])-set([np.array(angle).argmin()]))[1]])>90:
                                            initview=-initview
            

                                        # number of candidate axis
                                        num_candidate=6
                                        for i in range(num_candidate):
                                            D=-(planeview*plane_point).sum()
                                            newplane_fine.append(np.concatenate([rotate_vector(initview, planeview, i*np.pi/num_candidate),np.array([D])]))

                                        

                                        childfile.visual.face_colors=color_library[0]
                                        

                                        

                                        for i in range(num_candidate+1):
                                            arrow_mesh,rotation=create_3d_arrow_mesh(position=plane_point+0.04*newplane_fine[0][:3]/np.linalg.norm(newplane_fine[0][:3]),direction=newplane_fine[i][:3],arrow_length=2.2,color=color_library[i+2],stem_radius=0.01,head_radius=0.035)
                                            if i==0:
                                                combined=arrow_mesh
                                                
                                            else:
                                                combined = trimesh.util.concatenate([combined,arrow_mesh])

                                                
                                        newplane_fine.append(np.array([0,0,1,0]))
                                        newplane_fine.append(np.array([1,0,0,0]))
                                        newplane_fine.append(np.array([0,1,0,0]))

                                        final = trimesh.util.concatenate([childfile,othermesh,combined])
                                        final.export(os.path.join(savepath,'objs_mov',target+'-'+str(child)+'.obj'))

                                        #save the parameter of axis
                                        np.save(os.path.join(savepath,'objs_mov',target+'-'+str(child)+'.npy'),np.concatenate([plane_point[None],np.stack(newplane_fine)[:,:3]],0))
                                        
                                        logger.info('success :'+target)

                            # --- Generate candidate points and axes for C ---    
                            if nei_part['movement_type']=='C' and nei_part['child_label']==target_part:
                                num_mov+=1
                                parent=nei_part['parent_label']
                                child=nei_part['child_label']

                                if str(child) not in existpart:

                                    existpart.append(str(child))
                                    existpartpair.append(str(child)+'_'+str(parent))
                                    childfile=trimesh.load(os.path.join(partsegpath,target,'objs',str(child)+'.obj'),force='mesh')

                                    otherlist=os.listdir(os.path.join(partsegpath,target,'objs'))
                                    otherlist=list(set(otherlist)-set([str(child)+'.obj']))
                                    for other in range(len(otherlist)):
                                        if other==0:
                                            othermesh=trimesh.load(os.path.join(partsegpath,target,'objs',otherlist[other]),force='mesh')
                                        else:
                                            othermesh = trimesh.util.concatenate([othermesh,trimesh.load(os.path.join(partsegpath,target,'objs',otherlist[other]),force='mesh')])
                                    

                                    value,ind=((torch.Tensor(othermesh.vertices)[None].repeat(len(childfile.vertices),1,1)-torch.Tensor(childfile.vertices)[:,None].repeat(1,len(othermesh.vertices),1))**2).sum(-1).min(0)

                                    error=0
                                    threshold=1e-2
                                    jointregion=torch.Tensor(othermesh.vertices)[torch.where(value.sqrt()<threshold)]
                                    while len(jointregion)<=6:
                                        
                                        jointregion=torch.Tensor(othermesh.vertices)[torch.where(value.sqrt()<threshold)]
                                        threshold+=1e-2

                                        
                                    k = min(len(jointregion),5)  # number of groups
                                    kmeans = KMeans(n_clusters=k, random_state=42)
                                    kmeans.fit(jointregion)

                                    # center and label of groups
                                    labels = kmeans.labels_
                                    centers = kmeans.cluster_centers_

                                    
                                    for sph in range(len(centers)):
                                        if sph==0:
                                            sphere = trimesh.creation.icosphere(radius=0.03, subdivisions=3)
                                            
                                            sphere.apply_translation(centers[sph])
                                            sphere.visual.face_colors=color_library[sph+2]
                                            comsphere=sphere
                                        else:
                                            sphere = trimesh.creation.icosphere(radius=0.03, subdivisions=3)
                                            
                                            sphere.apply_translation(centers[sph])
                                            sphere.visual.face_colors=color_library[sph+2]
                                            comsphere = trimesh.util.concatenate([comsphere,sphere])


                                
                                    pcd = o3d.geometry.PointCloud()
                                    pcd.points = o3d.utility.Vector3dVector((jointregion).numpy())

                                    planes,innerpoint = fit_planes(pcd, distance_threshold=0.05, max_planes=1, min_inliers=int(len(jointregion)*0.1))
                                    
                                    para1,para2,para3,para4=planes[0]
                                    plane_point=jointregion[innerpoint[0]].numpy().mean(0)
                                    inner=[para1,para2,para3]*plane_point
                                    para4=-inner.sum()


                                    
                                    orilist=np.array([[1,0,0],[0,1,0],[0,0,1]])
                                    view=[]
                                    angle=[]
                                    for i in range(3):
                                        
                                        if vector_angle_np(planes[0][:3],orilist[i])>90:
                                            angle.append(180-vector_angle_np(planes[0][:3],orilist[i]))
                                            view.append(-planes[0][:3])
                                        else:
                                            angle.append(vector_angle_np(planes[0][:3],orilist[i]))
                                            view.append(planes[0][:3])
                                    planeview=view[np.array(angle).argmin()]    


                                    newplane_fine=[np.array([planeview[0],planeview[1],planeview[2],para4])]
                                    initview=np.cross(planeview,orilist[list(set([0,1,2])-set([np.array(angle).argmin()]))[0]])
                                    if vector_angle_np(initview,orilist[list(set([0,1,2])-set([np.array(angle).argmin()]))[1]])>90:
                                        initview=-initview
   
                                    num_candidate=6
                                    for i in range(num_candidate):
                                        D=-(planeview*plane_point).sum()
                                        newplane_fine.append(np.concatenate([rotate_vector(initview, planeview, i*np.pi/num_candidate),np.array([D])]))


                                    childfile.visual.face_colors=color_library[0]
                                    

                                    for i in range(len(newplane_fine)):
                                        arrow_mesh,rotation=create_3d_arrow_mesh_rotation(position=plane_point+0.04*newplane_fine[0][:3]/np.linalg.norm(newplane_fine[0][:3]),direction=newplane_fine[i][:3],arrow_length=3,color=color_library[i+2],stem_radius=0.01,head_radius=0.035)
                                        if i==0:
                                            combined=arrow_mesh
                                            newrotationarrow = trimesh.Trimesh(vertices=rotation_arrow.vertices, faces=rotation_arrow.faces)
                                            rotation_matrix = trimesh.geometry.align_vectors([1, 0, 0], newplane_fine[i][:3] )

                                           
                                            scale_factor = 0.3 / newrotationarrow.bounding_box.extents.max()
                                            transform = trimesh.transformations.scale_matrix(scale_factor)
                                            newrotationarrow.apply_transform(transform)
                                            newrotationarrow.apply_transform(rotation_matrix)

                                            newrotationarrow.apply_translation(plane_point+1.2*newplane_fine[i][:3]/np.linalg.norm(newplane_fine[i][:3]))
                                            newrotationarrow.visual.face_colors=color_library[i+2]
                                            combinedarrow=newrotationarrow
                                        else:
                                            combined = trimesh.util.concatenate([combined,arrow_mesh])

                                            newrotationarrow = trimesh.Trimesh(vertices=rotation_arrow.vertices, faces=rotation_arrow.faces)
                                            rotation_matrix = trimesh.geometry.align_vectors([1, 0, 0], newplane_fine[i][:3] )

                                           
                                            scale_factor = 0.3 / newrotationarrow.bounding_box.extents.max()
                                            transform = trimesh.transformations.scale_matrix(scale_factor)
                                            newrotationarrow.apply_transform(transform)
                                            newrotationarrow.apply_transform(rotation_matrix)
                                            newrotationarrow.apply_translation(plane_point+1.2*newplane_fine[i][:3]/np.linalg.norm(newplane_fine[i][:3]))
                                            newrotationarrow.visual.face_colors=color_library[i+2]
                                            combinedarrow = trimesh.util.concatenate([combinedarrow,newrotationarrow])
                                    

                                    newplane_fine.append(np.array([0,0,1,0]))
                                    newplane_fine.append(np.array([1,0,0,0]))
                                    newplane_fine.append(np.array([0,1,0,0]))   

                                    final = trimesh.util.concatenate([childfile,othermesh,comsphere])
               
                                    final.export(os.path.join(savepath,'objs_rotate',target+'-'+str(child)+'.obj'))

                                    final_axis = trimesh.util.concatenate([othermesh,childfile,comsphere,combined,combinedarrow])
                                    final_axis.export(os.path.join(savepath,'objs_axis',target+'-'+str(child)+'.obj'))
                                    np.save(os.path.join(savepath,'objs_rotate',target+'-'+str(child)+'.npy'),np.concatenate([plane_point[None],np.stack(newplane_fine)[:,:3],centers],0))
                                    
                                    logger.info('success :'+target)
                            

                            # --- Generate candidate points for D ---
                            if nei_part['movement_type']=='D' and nei_part['child_label']==target_part:  
                                num_mov+=1
                                parent=nei_part['parent_label']
                                child=nei_part['child_label']
                                
                                if str(child) not in existpart:
                                    existpart.append(str(child))
                                    existpartpair.append(str(child)+'_'+str(parent))
                                    childfile=trimesh.load(os.path.join(partsegpath,target,'objs',str(child)+'.obj'),force='mesh')

                                    otherlist=os.listdir(os.path.join(partsegpath,target,'objs'))
                                    otherlist=list(set(otherlist)-set([str(child)+'.obj']))
                                    for other in range(len(otherlist)):
                                        if other==0:
                                            othermesh=trimesh.load(os.path.join(partsegpath,target,'objs',otherlist[other]),force='mesh')
                                        else:
                                            othermesh = trimesh.util.concatenate([othermesh,trimesh.load(os.path.join(partsegpath,target,'objs',otherlist[other]),force='mesh')])
                                    

                                    value,ind=((torch.Tensor(othermesh.vertices)[None].repeat(len(childfile.vertices),1,1)-torch.Tensor(childfile.vertices)[:,None].repeat(1,len(othermesh.vertices),1))**2).sum(-1).min(0)

                                    error=0
                                    threshold=1e-2
                                    jointregion=torch.Tensor(othermesh.vertices)[torch.where(value.sqrt()<threshold)]
                                    while len(jointregion)<=6:
                                        
                                        jointregion=torch.Tensor(othermesh.vertices)[torch.where(value.sqrt()<threshold)]
                                        threshold+=1e-2


                                    plane_point=jointregion.numpy().mean(0)
                                    np.save(os.path.join(savepath,'objs_point',target+'-'+str(child)+'.npy'),plane_point[None])
                            
             
                            
