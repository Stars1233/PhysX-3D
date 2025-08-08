
import os
import numpy as np
import trimesh
import trimesh
import argparse
import logging
import json
from PIL import Image



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

def transfer_uv_and_texture(source_mesh, target_vertices):
    """
    Transfers UV coordinates and texture from source mesh to target mesh vertices.
    
    Args:
        source_mesh (trimesh.Trimesh): Source mesh with UV and texture.
        target_vertices (np.ndarray): Vertices of the target mesh.
    
    Returns:
        np.ndarray: UV coordinates for the target mesh.
    """
    # Make sure the original mesh has UV coordinates
    if not hasattr(source_mesh.visual, 'uv'):
        raise ValueError("Source mesh does not have UV coordinates.")
    
    uv_source = source_mesh.visual.uv
    faces_source = source_mesh.faces
    
    # Initialize the target UV array
    uv_target = np.zeros((len(target_vertices), 2))

    closest_points, distances, face_ids = source_mesh.nearest.on_surface(target_vertices)

    for i, vertex in enumerate(target_vertices):
        # Query the nearest surface point in the original mesh

        face_id = face_ids[i]
        
        # Get the vertex coordinates of the original triangle
        tri_vertices = source_mesh.vertices[faces_source[face_id]]
        
        # Calculate the center of gravity coordinates
        
        bary = trimesh.triangles.points_to_barycentric(tri_vertices[None], [closest_points[i]])

        u, v, w = bary[0]

        
        # Get the corresponding UV coordinates

        uv_tri = uv_source[faces_source[face_id]]

        uv_target[i] = np.clip(u * uv_tri[0] + v * uv_tri[1] + w * uv_tri[2],0.0,1.0)

    
    return uv_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--range", type=int, default=1000)
    args = parser.parse_args()
    
    mergemeshpath='./phy_dataset'

    logger = get_logger(os.path.join('exp_texture'+str(args.index)+'.log'),verbosity=1)

    logger.info('start')

    
    with open('finalindex.json','r') as f:
        res=json.load(f)


    finallist=os.listdir(mergemeshpath)
    
    finallist=finallist[args.index*args.range:(args.index+1)*args.range]

    for name in finallist:
        if not os.path.exists(os.path.join(mergemeshpath, name, 'model_tex.obj')):
            if name in list(res.keys()):
                os.makedirs(os.path.join(mergemeshpath,name), exist_ok=True)

                finalmesh=trimesh.load(os.path.join(mergemeshpath,name,'model.obj'),force='mesh')


                orimesh=trimesh.load(os.path.join(mergemeshpath,name,'model.obj'),force='mesh')


                rotation_matrix = trimesh.geometry.align_vectors([1, 0, 0], [-1, 0, 0])
                orimesh.apply_transform(rotation_matrix)

                scale_factor = 1 / orimesh.bounding_box.extents.max()
                transform = trimesh.transformations.scale_matrix(scale_factor)
                orimesh.apply_transform(transform)
                center=(orimesh.vertices.max(0)+orimesh.vertices.min(0))/2
                orimesh.apply_translation(-center)

                shapemesh=trimesh.load(os.path.join(res[name],'models','model_normalized.obj'),force='mesh')
                


                scale_factor = 1 / shapemesh.bounding_box.extents.max()
                transform = trimesh.transformations.scale_matrix(scale_factor)
                shapemesh.apply_transform(transform)
                center=(shapemesh.vertices.max(0)+shapemesh.vertices.min(0))/2
                shapemesh.apply_translation(-center)
                
                if isinstance(shapemesh.visual,trimesh.visual.ColorVisuals):
                    target_uv=np.zeros((len(orimesh.vertices), 2))+0.5
                    refimg=Image.fromarray(np.uint8(np.zeros((1,1,3))+128))
                else:

                    if shapemesh.visual.uv is not None:
                        target_uv=transfer_uv_and_texture(shapemesh,orimesh.vertices)
                        if isinstance(shapemesh.visual.material,trimesh.visual.material.SimpleMaterial):
                            refimg=shapemesh.visual.material.to_pbr().baseColorTexture
                        elif isinstance(shapemesh.visual.material,trimesh.visual.material.PBRMaterial):
                            refimg=shapemesh.visual.material.baseColorTexture

                    else:
                        target_uv=np.zeros((len(orimesh.vertices), 2))+0.5
                        refimg=Image.fromarray(np.uint8(np.zeros((1,1,3))+128))
                target_mesh = trimesh.Trimesh(
                    vertices=finalmesh.vertices,
                    faces=finalmesh.faces,
                    visual=trimesh.visual.TextureVisuals(
                        uv=target_uv,
                        image=refimg
                    )
                )
                
                target_mesh.export(os.path.join(mergemeshpath, name, 'model_tex.obj'))
                
                    
                logger.info(name+' mapping')
            else:
                finalmesh=trimesh.load(os.path.join(mergemeshpath,name,'model.obj'),force='mesh')
                target_uv=np.zeros((len(finalmesh.vertices), 2))+0.5
                refimg=Image.fromarray(np.uint8(np.zeros((1,1,3))+128))

                target_mesh = trimesh.Trimesh(
                    vertices=finalmesh.vertices,
                    faces=finalmesh.faces,
                    visual=trimesh.visual.TextureVisuals(
                        uv=target_uv,
                        image=refimg
                    )
                )
                target_mesh.export(os.path.join(mergemeshpath, name, 'model_tex.obj'))


                logger.info(name+' notexture')
        
        else:
            logger.info(name+' skip')
           
