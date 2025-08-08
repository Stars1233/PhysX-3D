import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.dist_utils import read_file_dist
from ..utils.data_utils import load_balanced_group_indices


class SLatVisMixin:
    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        latent_model_phy: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.slat_dec = None
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        self.latent_model_phy=latent_model_phy
        
    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return

        cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
        decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
        
        ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_step{self.slat_dec_ckpt.zfill(7)}.pt')

        decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))

        propertydecoder = getattr(models, cfg['models']['property_decoder']['name'])(**cfg['models']['property_decoder']['args'])
        ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'property_decoder_step{self.slat_dec_ckpt.zfill(7)}.pt')
        propertydecoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))

        propertyoutput = getattr(models, cfg['models']['property_output']['name'])(**cfg['models']['property_output']['args'])
        ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'property_output_step{self.slat_dec_ckpt.zfill(7)}.pt')
        propertyoutput.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        
        self.slat_dec = decoder.cuda().eval()
        self.slat_dec_phy = propertydecoder.cuda().eval()
        self.slat_dec_phyout = propertyoutput.cuda().eval()

    def _delete_slat_dec(self):
        del self.slat_dec
        self.slat_dec = None
        del self.slat_dec_phy
        self.slat_dec_phy = None

    @torch.no_grad()
    def decode_latent(self, z,z_phy, batch_size=1):
        self._loading_slat_dec()
        reps = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
            z_phy = z_phy * self.std_phy.to(z_phy.device) + self.mean_phy.to(z_phy.device)
        for i in range(0, z.shape[0], batch_size):
            
            phy_property,hs = self.slat_dec_phy(z_phy[i:i+batch_size])

            reps.append(self.slat_dec(z[i:i+batch_size],phy_property,hs))
        reps = sum(reps, [])
        self._delete_slat_dec()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        x_0_phy = x_0['x_0_phy']
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0['x_0']

        
        reps = self.decode_latent(x_0.cuda(),x_0_phy.cuda())
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        renderer = get_renderer(reps[0])
        images = []
        masks=[]
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            mask = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr,return_types=['color','phy_map'])
                lang_map,phy_map=self.slat_dec_phyout(None,res['phy_map'].unsqueeze(0))
                mask[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)]=((torch.relu(phy_map[:,1])>0.85)&(torch.relu(phy_map[:,1])<1)).float()[:,None].repeat(1,3,1,1)[0]


                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            masks.append(mask)

        del self.slat_dec_phyout
        self.slat_dec_phyout = None

        images = torch.stack(images)
        masks = torch.stack(masks)

            
        return {
            "images":images,
            "masks":masks,
        }
    
    
class SLat(SLatVisMixin, StandardDatasetBase):
    """
    structured latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        latent_model_phy: Optional[str] = None,
    ):
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
            latent_model_phy=latent_model_phy,
        )
        self.latent_model_phy=latent_model_phy
        self.loads = [self.metadata.loc[sha256, 'num_voxels'] for _, sha256 in self.instances]
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
            self.mean_phy = torch.tensor(self.normalization['mean_phy']).reshape(1, -1)
            self.std_phy = torch.tensor(self.normalization['std_phy']).reshape(1, -1)
      
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))
        phy = np.load(os.path.join(root, 'latents', self.latent_model_phy, f'{instance}.npz')) 
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        phyfeats = torch.tensor(np.float32(phy['feats'])).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
            phyfeats = (phyfeats - self.mean_phy) / self.std_phy
        return {
            'coords': coords,
            'feats': feats,
            'phyfeats': phyfeats,
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            phyfeats=[]
            layout = []
            layout_phy = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                phyfeats.append(b['phyfeats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                layout_phy.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            phyfeats = torch.cat(phyfeats)
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0_phy'] = SparseTensor(
                coords=coords,
                feats=phyfeats,
            )

            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)

            pack['x_0_phy']._shape = torch.Size([len(group), *sub_batch[0]['phyfeats'].shape[1:]])
            pack['x_0_phy'].register_spatial_cache('layout_phy', layout_phy)
            
            # collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats','phyfeats']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs
        
    
class TextConditionedSLat(TextConditionedMixin, SLat):
    """
    Text conditioned structured latent dataset
    """
    pass


class ImageConditionedSLat(ImageConditionedMixin, SLat):
    """
    Image conditioned structured latent dataset
    """
    pass
