import os
import pandas as pd

import numpy as np
import json
savepath='../datasets/PhysXNet'
finaldataset='./phy_dataset'

os.makedirs(os.path.join(savepath), exist_ok=True)
os.makedirs(os.path.join(savepath,'merged_records'), exist_ok=True)
pathlist=[]
namelist=os.listdir(finaldataset)
namelist = sorted(namelist, key=lambda x: int(x))
for i in namelist:
    pathlist.append(os.path.join(finaldataset,i,'model_tex.obj'))

namelist_=[]
for i in namelist:
    namelist_.append(i+'_')
zero=np.zeros((len(namelist_))).tolist()
ten=(np.zeros((len(namelist_)))+10).tolist()
false=(np.zeros((len(namelist_)))!=0).tolist()

frame = pd.DataFrame({'sha256': namelist_, 'file_identifier': zero,'aesthetic_score': ten,'captions':zero,'rendered':false,'voxelized':false,'num_voxels': zero, 'cond_rendered': false,'local_path':pathlist},dtype='object')

frame.to_csv(os.path.join(savepath,'metadata.csv'), index=False, sep=',')

frame = pd.DataFrame({'sha256': namelist_, 'local_path':pathlist},dtype='object')

frame.to_csv(os.path.join(savepath,'merged_records','1743666055_downloaded_0.csv'), index=False, sep=',')