
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import trimesh

import base64
# Function to encode the image
import imageio
import glob
import cv2
import trimesh
import argparse
import logging
import json
import pandas as pd
import clip
import torch.nn.functional as F
import math



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--range", type=int, default=1)
    parser.add_argument("--datapath", type=str, default='./physxnet')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", jit=False, device=device)
    model=model.eval()


    fianljson=os.path.join(args.datapath,'finaljson') # physxnet path
    partsegpath=os.path.join(args.datapath,'partseg') 

    savepath='./phy_dataset/'
    
    namelist=os.listdir(fianljson)
    namelist=namelist[args.index*args.range:(args.index+1)*args.range]

    os.makedirs(savepath, exist_ok=True)
    logger = get_logger(os.path.join('merge'+str(args.index)+'.log'),verbosity=1)

    logger.info('start')

    existfile=os.listdir(savepath)
    
    for name in namelist:
        nextname=0
        name=name[:-5]
        logger.info('begin: '+name)
        replace=0
        

        if not os.path.exists(os.path.join(savepath,name,'model.obj')):
            
        
            os.makedirs(os.path.join(savepath,name), exist_ok=True)
            jsonfile=os.path.join(fianljson,name+'.json')
            with open(jsonfile,'r') as f:
                data=json.load(f)

            namepath=os.path.join(partsegpath,name,'objs')
            objlist = sorted(os.listdir(namepath), key=lambda x: int(x.split('.')[0]))
            allinfo_emb=[]
            allinfo_index=[]
            oldallinfo_emb=[]
            rangenum=[]
            otherproperty=[]
            otherproperty1=[]
            for objname in range(len(objlist)):


                groupnum=len(data['group_info'])
                exist=0
                for groupind in range(groupnum):
                    if groupind==0:
                        checklist=data['group_info'][str(groupind)]
                    else:
                        checklist=data['group_info'][str(groupind)][0]
                    
                    if objname in checklist and groupind!=0:
                        groupnumber=groupind
                        parentnumber=int(data['group_info'][str(groupind)][1])
                        movpro=data['group_info'][str(groupind)][2]

                        if data['group_info'][str(groupind)][3]=='A':
                            movement_type=4
                        if data['group_info'][str(groupind)][3]=='B':
                            movement_type=3
                        if data['group_info'][str(groupind)][3]=='C' or data['group_info'][str(groupind)][3]=='CB':
                            movement_type=2
                        if data['group_info'][str(groupind)][3]=='D':
                            movement_type=1
                        if data['group_info'][str(groupind)][3]=='E':
                            movement_type=0
                        exist+=1

                    elif objname in checklist and groupind==0:
                        groupnumber=groupind
                        parentnumber=-1
                        movpro=np.zeros((8))-1
                        movement_type=0
                        exist+=1
                if exist>1:
                    logger.info('error_repeat: '+name)
                if nextname:
                    break






                
                partinfo=data['parts'][objname]
                partinfo['dimension_scale']=data['dimension']
                newrotationarrow=trimesh.load(os.path.join(namepath,objlist[objname]),force='mesh')
                str_list=partinfo['dimension_scale'].split(' ')[0].split('*')
                sorted_list = sorted(str_list, key=float, reverse=True)
                rank=int(partinfo['priority_rank'])
                scaling=float(sorted_list[0])
                density=float(partinfo['density'].split(' ')[0])


                otherproperty1.append(np.concatenate([[scaling],[rank],[density],[groupnumber],[parentnumber],movpro,[movement_type]])[None])


                tokens1 = clip.tokenize(partinfo['Basic_description']).to(device)
                tokens2 = clip.tokenize(partinfo['Functional_description']).to(device)
                tokens3 = clip.tokenize(partinfo['Movement_description']).to(device)

                info_index=(torch.zeros(1)+objname).repeat(len(newrotationarrow.vertices))
                info_emb=torch.cat([model.encode_text(tokens1),model.encode_text(tokens2),model.encode_text(tokens3),model.encode_text(tokens3)],0)[None,:,:]
                allinfo_emb.append(info_emb)
                allinfo_index.append(info_index)

                rangenum.append(len(newrotationarrow.vertices))
                if objname==0:
                    combinedarrow=newrotationarrow
                    if objname==len(objlist)-1:
                        finalind=torch.cat(allinfo_index).float()
                else:
                    combinedarrow = trimesh.util.concatenate([combinedarrow,newrotationarrow])
                    
                    if objname==len(objlist)-1:
                        aa = combinedarrow.copy()
                        combinedarrow.merge_vertices()
                        if len(aa.vertices)!=len(combinedarrow.vertices):
                            ind=np.zeros((len(combinedarrow.vertices)))
                            for chunk in range(math.ceil(len(combinedarrow.vertices)/10000)):
                                ind[chunk*10000:(chunk+1)*10000]=((torch.Tensor(aa.vertices)[:,None,:].repeat(1,len(ind[chunk*10000:(chunk+1)*10000]),1)-torch.Tensor(combinedarrow.vertices)[None,:,:][:,chunk*10000:(chunk+1)*10000].repeat(len(aa.vertices),1,1))**2).sum(-1).argmin(0).numpy()
                            finalind=torch.cat(allinfo_index)[ind].float()
                        else:
                            finalind=torch.cat(allinfo_index).float()


            finalres=torch.cat(allinfo_emb).float()
            finalotherproperty=np.concatenate(otherproperty1)
            np.save(os.path.join(savepath,name,'clip.npy'),finalres.cpu().detach().numpy())
            np.save(os.path.join(savepath,name,'clip_ind_new.npy'),finalind.numpy())
            np.save(os.path.join(savepath,name,'otherproperty.npy'),finalotherproperty)
            combinedarrow.export(os.path.join(savepath,name,'model.obj'))
            logger.info('success: '+name)
        else:
            logger.info('skip: '+name)

