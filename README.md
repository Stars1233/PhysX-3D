<div align="left">
<h1 align="center">PhysX-3D: Physical-Grounded 3D Asset Generation
</h1>


<p align="center"><a href="https://arxiv.org/abs/2507.12465"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://physx-3d.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=homepage&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/Caoza/PhysX'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
<a href='https://youtu.be/M5V_c0Duuy4'><img src='https://img.shields.io/youtube/views/M5V_c0Duuy4'></a>

<div align="center">
    <a href="https://ziangcao0312.github.io/" target="_blank">Ziang Cao</a><sup>1</sup>,
    <a href="https://frozenburning.github.io/" target="_blank">Zhaoxi Chen</a><sup>1</sup>,
    <a href="https://github.com/paul007pl" target="_blank">Liang Pan</a><sup>2</sup>,
    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>1</sup>
</div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp; <sup>2</sup>Shanghai AI Laboratory
</div>
<div>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="img/teaser.png">
</div>

<strong>PhysX provides a new end-to-end paradigm for physical-grounded 3D asset generation.</strong>

:open_book: For more visual results, go checkout our <a href="https://physx-3d.github.io/" target="_blank">project page</a>

## PhysXNet & PhysXNet-XL

For more details about our proposed dataset, please see this [link](https://huggingface.co/datasets/Caoza/PhysX-3D)

The scripts for annotation and obtaining texture information are located in `./tools`.

## PhysGen 

### Training

1. Download and preprocess the PhysXNet Dataset

```bash
huggingface-cli download Caoza/PhysX-3D PhysXNet.zip --repo-type dataset --local-dir ./dataset_toolkits/
cd dataset_toolkits
unzip PhysXNet.zip -d physxnet
```

**Note:**  Since [PartNet](https://huggingface.co/datasets/ShapeNet/PartNet-archive) has no texture information, you need to download the [ShapeNet](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) dataset and save it to `./dataset_toolkits/shapenet` to obtain texture information. 

```bash
bash precess.sh
```

2. VAE training

```python
python train.py 
     --config configs/vae/slat_vae_enc_dec_mesh_phy.json 
     --output_dir outputs/vae_phy 
     --data_dir ./datasets/PhysXGen 
```

3. Diffusion training

```python
python train.py 
     --config configs/generation/slat_flow_img_dit_L_phy.json 
     --output_dir outputs/diffusion_phy 
     --data_dir ./datasets/PhysXGen 
```

### Inference

1. Download the pre-train model from huggingface.

```bash
bash download_pretrain.sh
```

2. Run the inference code

```bash
python example.py
```

## References

If you find PhysX useful for your work please cite:
```
@article{cao2025physx-3d,
  title={PhysX-3D: Physical-Grounded 3D Asset Generation},
  author={Cao, Ziang and Chen, Zhaoxi and Pan, Liang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2507.12465},
  year={2025}
}
```
### Acknowledgement

The data and code is based on [PartNet](https://huggingface.co/datasets/ShapeNet/PartNet-archive) and [TRELLIS](https://github.com/microsoft/TRELLIS). We would like to express our sincere thanks to the contributors.

## :newspaper_roll: License

Distributed under the S-Lab License. See `LICENSE` for more information.

<div align="center">
  <a href="https://info.flagcounter.com/ukVw"><img src="https://s01.flagcounter.com/map/ukVw/size_s/txt_000000/border_CCCCCC/pageviews_0/viewers_0/flags_0/" alt="Flag Counter" border="0"></a>
</div>
