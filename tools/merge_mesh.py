
import os
import trimesh
import argparse
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--range", type=int, default=1000)
    args = parser.parse_args()



    fianljson='./physxnet/finaljson'
    partsegpath='./physxnet/partseg'

    savepath='./phy_dataset/'
    
    namelist=os.listdir(fianljson)
    namelist=namelist[args.index*args.range:(args.index+1)*args.range]

    os.makedirs(savepath, exist_ok=True)
    logger = get_logger(os.path.join('./output_physxnet','exp_merge'+str(args.index)+'.log'),verbosity=1)

    logger.info('start')

    existfile=os.listdir(savepath)

    for name in namelist:
        name=name[:-5]
        logger.info('begin: '+name)
        
        if not os.path.exists(os.path.join(savepath,name,'model.obj')):
            

            namepath=os.path.join(partsegpath,name,'objs')
            objlist = sorted(os.listdir(namepath), key=lambda x: int(x.split('.')[0]))

            for objname in range(len(objlist)):

                newrotationarrow=trimesh.load(os.path.join(namepath,objlist[objname]),force='mesh')
                
                if objname==0:
                    combinedarrow=newrotationarrow
                    
                else:
                    combinedarrow = trimesh.util.concatenate([combinedarrow,newrotationarrow])
                    combinedarrow.merge_vertices()
                        

            combinedarrow.export(os.path.join(savepath,name,'model.obj'))
            logger.info('success: '+name)
        else:
            logger.info('skip: '+name)
