# Annotation pipeline & Texture retrieval



## Annotation pipeline

### 1_gptanno.py 

```
python gptanno.py
```

Get GPT-4o output results through the API and save them as JSON file

### 2_vis_kinematic.py 

```
python 2_vis_kinematic.py 
```

According to the preliminary results of the gpt annotation, the candidate axes and candidate points of the translation (B), rotation (C), and articulation (D) parts are saved as mesh

**Note**: arrow_norm.obj is used for better visualization

### 3_human_in_the_loop annotation and check

Manually select the candidate axes and points for B, C, and D



## Texture retrieval

Since [PartNet](https://huggingface.co/datasets/ShapeNet/PartNet-archive) has no texture information, you need to download the [ShapeNet](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) dataset and save it to `./shapenet` to obtain texture information. 

## merge_mesh.py 

```
python merge_mesh.py 
```

Merge the parts into one mesh in order to calculate the texture correspondence with the original mesh

## retrieval_texture_example.py 

```
python retrieval_texture_example.py 
```

Since the original meshes in ShapeNet are similar in shape to the meshes in PartNet, the nearest texture information can be obtained based on their coordinate correspondence.

**Note**: finalindex.json is obtain from the metadata of [PartNet](https://huggingface.co/datasets/ShapeNet/PartNet-archive)
