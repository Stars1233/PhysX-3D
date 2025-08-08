import os
from PIL import Image
import json
import numpy as np
import torch
import utils3d.torch
from ..modules.sparse.basic import SparseTensor
from .components import StandardDatasetBase

import cv2
import imageio
import os
import numpy as np
from io import BytesIO




class SLat2Rendermesh(StandardDatasetBase):
    """
    Dataset for Structured Latent and rendered images.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        latent_model (str): latent model name
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
    ):
        self.image_size = image_size
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)

        
        super().__init__(roots)
        
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats

    def _get_image(self, root, instance):
        with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(root, 'renders', instance, metadata['file_path'])

        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
    
    def _get_latent(self, root, instance):
        data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        return {
            'coords': coords,
            'feats': feats,
        }
    def _get_latent_phy(self, root, instance):
        data = np.load(os.path.join(root,'latents','dinov2_vitl14_reg_physxgen_100000', f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        return {
            'feats_phy': feats,
        }
    def _get_property(self, root, instance):

        property1 = torch.tensor(np.float32(np.load(os.path.join(root, 'property', instance+'1.npy')))).float()
        property1=property1.reshape(len(property1),-1)
        property2 = torch.tensor(np.float32(np.load(os.path.join(root, 'property', instance+'2.npy')))).float()

        return {
            'property1': torch.tensor(property1),
            'property2': torch.tensor(property2),
        }

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        pack = {}
        coords = []
        oricoords = []
        for i, b in enumerate(batch):
            coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
       

        property1 = torch.cat([b['property1'] for b in batch])
        property2 = torch.cat([b['property2'] for b in batch])
        pack['latents'] = SparseTensor(
            coords=coords,
            feats=feats,
        )

       
        pack['property1'] = SparseTensor(
            coords=coords,
            feats=property1,
        )
        pack['property2'] = SparseTensor(
            coords=coords,
            feats=property2,
        )

        
        keys = [k for k in batch[0].keys() if k not in ['coords', 'feats', 'feats_phy','property1','property2']]
        for k in keys:
            if isinstance(batch[0][k], torch.Tensor):
                pack[k] = torch.stack([b[k] for b in batch])
            elif isinstance(batch[0][k], list):
                pack[k] = sum([b[k] for b in batch], [])
            else:
                pack[k] = [b[k] for b in batch]

        return pack

    def get_instance(self, root, instance):
        image = self._get_image(root, instance)
        latent = self._get_latent(root, instance)
        return {
            **image,
            **latent,
        }
        
       
class Slat2RenderGeomesh(SLat2Rendermesh):
    def __init__(
        self,
        roots: str,
        image_size: int,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
    ):
        super().__init__(
            roots,
            image_size,
            latent_model,
            min_aesthetic_score,
            max_num_voxels,
        )
        
    def _get_geo(self, root, instance):
        verts, face = utils3d.io.read_ply(os.path.join(root, 'renders', instance, 'mesh_new.ply'))
        ind=np.int32(np.load(os.path.join('./dataset_toolkits/phy_dataset', instance[:-1], 'clip_ind_new.npy')))
        property1=np.float32(np.load(os.path.join('./dataset_toolkits/phy_dataset', instance[:-1], 'clip.npy'))[ind])
        property2=np.float32(np.load(os.path.join('./dataset_toolkits/phy_dataset', instance[:-1], 'otherproperty.npy'))[ind])
        
        
        mesh = {
            "vertices" : torch.from_numpy(verts),
            "faces" : torch.from_numpy(np.int32(face)),
            "property1" : torch.from_numpy((property1)),
            "property2" : torch.from_numpy((property2)),
        }
        return  {
            "mesh" : mesh,
        }
    def _get_feat(self, root, instance):
        DATA_RESOLUTION = 64
        feats_path = os.path.join(root, 'features', 'dinov2_vitl14_reg', f'{instance}.npz')
        feats = np.load(feats_path, allow_pickle=True)
        coords = torch.tensor(feats['indices']).int()
        feats = torch.tensor(feats['patchtokens']).float()
        
        if 64 != DATA_RESOLUTION:
            factor = DATA_RESOLUTION // 64
            coords = coords // factor
            coords, idx = coords.unique(return_inverse=True, dim=0)
            feats = torch.scatter_reduce(
                torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
                dim=0,
                index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
                src=feats,
                reduce='mean'
            )
        
        feat= {
            'oricoords': coords,
            'orifeats': feats,
        }
        return feat
        
    def get_instance(self, root, instance):
        image = self._get_image(root, instance)
        geo = self._get_geo(root, instance)
        latent = self._get_latent(root, instance)
        properties = self._get_property(root, instance)
        return {
            **image,
            **latent,
            **geo,
            **properties,
        }
        
        