
import os
import numpy as np
import base64
from openai import OpenAI
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
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--range", type=int, default=40000)
    args = parser.parse_args()

    
    save_dir='./output_physxnet'
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    os.makedirs(os.path.join(save_dir,'1_gpt_annotation'), exist_ok=True)

    input_pc_file='./physxnet/partseg'
    filelist=os.listdir(input_pc_file)
    filelist=filelist[args.index*args.range:(args.index+1)*args.range]

    logger = get_logger(os.path.join(save_dir,'exp_1gpt'+str(args.index)+'.log'),verbosity=1)

    logger.info('start')


    expense=0
    for filename in filelist:

            logger.info('begin: '+filename)

            if os.path.exists(os.path.join(save_dir,'1_gpt_annotation',filename+'.json')):
                logger.info('finish: '+filename)

            else:
                client = OpenAI()
                imgpath=os.path.join(input_pc_file,filename,'imgs')
                num_part=len(os.listdir(imgpath))
                partname=''
                sorted_list = sorted(os.listdir(imgpath), key=lambda x: int(x.split('_')[0]))

                for name in sorted_list:
                    if name==sorted_list[-1]:
                        partname=partname+'and '+name[:-4]
                    else:
                        partname=partname+name[:-4]+', '
                
                system='''You have a good understanding of the structure of an articulated object. Your job is to assist the user in analyzing the properties of it. Specifically, the user will give you images of parts, and your task is to recognize the articulated object and analyze the parts of that object. You should find a similar physical 3D object in the real world. Based on human knowledge of it, you should give your answer about the information as follows:
                        Object-level: 
                        (1) name, category, and dimension (length*width*height, in cm) of the articulated object.

                        Part-level: 
                        Part_1 (image_1):
                        (1) Label, name, material, density (g/cm^3) of the part.
                        (2) priority rank of being touched when using this object based on human preference.
                        (3) labels of all neighboring parts.
                        (3.1) assign a movement type for each group between Part_1 and its neighboring parts (A. merely touch and no movement constraints, B. relative translationally move, C. rotation about an axis, D. rotation about a point, or E. rigid constraint). If the movement type is B, C, or D, output the parent and child parts.
                        (3.2) assign a movement type for each group between Part_1 and its neighboring parts (A. merely touch and no movement constraints, B. relative translationally move, C. rotation about an axis, D. rotation about a point, or E. rigid constraint). If the movement type is B, C, or D, output the parent and child parts.
                        ...
                        (4) summarize the basic information (including material, physical dimension, category, and name), functional, movement description, and priority of being grasped description.
                        Part_2 (image_2):
                        (1) Label, name, material, density (g/cm^3) of the part.
                        (2) priority rank of being touched when using this object based on human preference.
                        (3) labels of all neighboring parts.
                        (3.1) assign a movement type for each group between Part_2 and its neighboring parts (A. merely touch and no movement constraints, B. relative translationally move, C. rotation about an axis, D. rotation about a point, or E. rigid constraint). If the movement type is B, C, or D, output the parent and child parts.
                        (3.2) assign a movement type for each group between Part_2 and its neighboring parts (A. merely touch and no movement constraints, B. relative translationally move, C. rotation about an axis, D. rotation about a point, or E. rigid constraint). If the movement type is B, C, or D, output the parent and child parts.
                        ...
                        (4) summarize the basic information (including material, physical dimension, category, and name), functional, movement description, and priority of being grasped description.

                        For example:
                        {
                        "object_name": "Rifle",
                        "category": "ToyGun",
                        "dimension": "80*10*25",
                        "parts": [
                            {
                            "label": 1,
                            "material": "Plastic",
                            "density": "1.2 g/cm^3",
                            "name": "Foregrip",
                            "priority_rank": 2,
                            "neighbors": [
                                {
                                "labels_of_movement_group": "1-8",
                                "movement_type": "E",
                                }
                            "Basic_description": "It's a foregrip of a Rifle made of plastic.",
                            "Functional_description": "It can control the ...",
                            "Movement_description": "It cannot move normally...",
                            "Grasped_description": "Most likely to be grasped or handled.",
                            ]
                            },
                            {
                            "label": 2,
                            "material": "Plastic",
                            "density": "1.2 g/cm^3",
                            "name": "Stock",
                            "priority_rank": 5,
                            "neighbors": [
                                {
                                "labels_of_movement_group": "2-8",
                                "movement_type": "B",
                                "parent_label": 8,
                                "child_label": 2
                                }
                            "Basic_description": "It's a foregrip of a Rifle classified as a gun. It is a big part of the object made of plastic.",
                            "Functional_description": "It can be grasped to control the object...",
                            "Movement_description": "It cannot move normally...",
                            "Grasped_description": "Less likely to be grasped.",
                            ]
                            },
                            ...
                        }

                        Remember:
                        (1) Do not answer anything not asked.
                        (2) You should base on the physical 3D object in the real world to analyze the properties and movement of the object.
                        (3) You should purely based on its function to detremine the movement type of parts.
                        (4) You should prefer to analyze the rendered object as a real 3D object rather than a toy model.
                        (5) You should assign the priority rank of being grasped from 1 to 10. The most likely part to be touched is 1.
                        (6) You should consider the function rather than the area or name of the target part to determine the priority rank of being grasped.
                        (7) The target part uses red color while the other parts use grey color.
                        (8) You should output full JSON including all parts. 
                        '''
                old="Analyze the "+str(num_part)+" parts of a 3D object. Each image includes one part. From the first to last image part names are"+partname+". Just output the object-level and part-level information for this object in JSON."
                
                content=[
                            {
                                "type": "text",
                                "text": old,
                            },      
                        ]

                for name in sorted_list:
            
                    meshpath=os.path.join(imgpath,name)
                    base64_image=encode_image(meshpath)
                    content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                                },
                        )

                response = client.chat.completions.create(
                    model="chatgpt-4o-latest",    

                    messages=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": system,
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    temperature=0,
                    
                )
                
                
                with open(os.path.join(save_dir,'1_gpt_annotation',filename+'.json'),'w') as file:
                    file.write( response.choices[0].message.content[8:-4])
                

                expense=expense+15*response.usage.completion_tokens/1000000+5*response.usage.prompt_tokens/1000000
                logger.info('total_cost: '+str(expense))  
                logger.info('success: '+filename)

                