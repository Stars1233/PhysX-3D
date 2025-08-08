import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import trellis.models as models
import trellis.modules.sparse as sp
import cv2
import imageio
import os
import numpy as np
from io import BytesIO

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--feat_model', type=str, default='dinov2_vitl14_reg',
                        help='Feature model')
    parser.add_argument('--pretrained_path', type=str, default='./pretrain/vae',
                        help='Pretrained encoder model')
    parser.add_argument('--enc_model', type=str, default='physxgen',
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default='100000',
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    
    latent_name = f'{opt.feat_model}_{opt.enc_model}_{opt.ckpt}'
    cfg = edict(json.load(open(os.path.join(opt.pretrained_path, 'config.json'), 'r')))
    property_encoder = getattr(models, cfg.models.property_encoder.name)(**cfg.models.property_encoder.args).cuda()
    ckpt_path = os.path.join(opt.pretrained_path,'ckpts','property_encoder_step0100000.pt')
    property_encoder.load_state_dict(torch.load(ckpt_path), strict=False)
    property_encoder.eval()
    print(f'Loaded model from {ckpt_path}')
    
    os.makedirs(os.path.join(opt.output_dir, 'latents', latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt.instances is not None:
        with open(opt.instances, 'r') as f:
            sha256s = [line.strip() for line in f]
        metadata = metadata[metadata['sha256'].isin(sha256s)]
    else:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata[f'feature_{opt.feat_model}'] == True]
        if f'latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'latent_{latent_name}'] == False]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'latent_{latent_name}': True})
            sha256s.remove(sha256)

    

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=1) as loader_executor, \
            ThreadPoolExecutor(max_workers=1) as saver_executor:
            def loader(sha256):
                try:
                    feats = np.load(os.path.join('../datasets/PhysXNet/', 'features', opt.feat_model, f'{sha256}.npz'))
                    realfeats1=np.float32(np.load(os.path.join('../datasets/PhysXNet/', 'property', f'{sha256}1.npy')))
                    realfeats2=np.float32(np.load(os.path.join('../datasets/PhysXNet/', 'property', f'{sha256}2.npy')))
                    load_queue.put((sha256, feats,realfeats1,realfeats2))
                except Exception as e:
                    print(f"Error loading features for {sha256}: {e}")
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                save_path = os.path.join(opt.output_dir, 'latents', latent_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'latent_{latent_name}': True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, feats,realfeats1,realfeats2 = load_queue.get()

                feats1 = sp.SparseTensor(
                    feats = torch.from_numpy(realfeats1).reshape(len(realfeats1),-1).float(),
                    coords = torch.cat([
                        torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                        torch.from_numpy(feats['indices']).int(),
                    ], dim=1),
                ).cuda()
                feats2 = sp.SparseTensor(
                    feats = torch.from_numpy(realfeats2).float(),
                    coords = torch.cat([
                        torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                        torch.from_numpy(feats['indices']).int(),
                    ], dim=1),
                ).cuda()
                
                latent = property_encoder(feats1,feats2, sample_posterior=True)
                assert torch.isfinite(latent.feats).all(), "Non-finite latent"
                pack = {
                    'feats': latent.feats.cpu().numpy().astype(np.float32),
                    'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                saver_executor.submit(saver, sha256, pack)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'latent_{latent_name}_{opt.rank}.csv'), index=False)

