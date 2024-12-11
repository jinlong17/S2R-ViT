<!--
 * @Descripttion: 
 * @version: 
 * @Author: Jinlong Li CSU PhD
 * @Date: 2024-07-11 13:51:21
 * @LastEditors: Jinlong Li CSU PhD
 * @LastEditTime: 2024-12-11 15:51:08
-->
# S2R-ViT for multi-agent cooperative perception: Bridging the gap from simulation to reality
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2303.07601-b31b1b.svg)](https://arxiv.org/pdf/2303.07601.pdf) 


This is the official implementation of ICRA 2042 paper. "S2R-ViT for multi-agent cooperative perception: Bridging the gap from simulation to reality".
[Jinlong Li](https://jinlong17.github.io/),
[Runsheng Xu](https://derrickxunu.github.io/),
[Xinyu Liu](https://scholar.google.com/citations?user=fGK5P7IAAAAJ&hl=zh-CN),
[Baolu Li](https://scholar.google.com/citations?user=d94_GW4AAAAJ&hl=en),
[Qin Zou](https://scholar.google.com/citations?user=dJ8izFAAAAAJ&hl=en),
[Jiaqi Ma](https://mobility-lab.seas.ucla.edu/),
[Hongkai Yu](https://scholar.google.com/citations?user=JnQts0kAAAAJ&hl=en).


IEEE International Conference on Robotics and Automation (ICRA) 2024!

## [Project Page](https://jinlong17.github.io/S2R-ViT/)

<p align="center">
  <img src="./imgs/S2R-ViT.png" alt="teaser" width="90%" height="90%">
</p>

 
## S2R-UViT: Simulation-to-Reality Uncertainty-aware Vision Transformer


<p align="center">
  <img src="./imgs/S2R-UViT.png" alt="teaser" width="75%" height="75%">
</p>

 
## Data  Download

We conduct experiments on two public benchmark datasets (OPV2V, V2V4Real) for the V2V cooperative perception task You can download these data from [OPV2V](https://github.com/DerrickXuNu/OpenCOOD) and [V2V4Real](https://github.com/ucla-mobility/V2V4Real).


## Getting Started

### Environment Setup

S2R-ViT's codebase is build upon [V2V4Real](https://github.com/ucla-mobility/V2V4Real). this codebase supports both the simulation and real-world data and more perception tasks.
To set up the codebase environment(following the [V2V4Real](https://github.com/ucla-mobility/V2V4Real)), do the following steps:
#### 1. Create conda environment (python >= 3.7)
```shell
conda create -n v2v4real python=3.7
conda activate v2v4real
```
#### 2. Pytorch Installation (>= 1.12.0 Required)
Take pytorch 1.12.0 as an example:
```shell
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
#### 3. spconv 2.x Installation
```shell
pip install spconv-cu113
```
#### 4. Install other dependencies
```shell
pip install -r requirements.txt
python setup.py develop
```
#### 5.Install bbx nms calculation cuda version
```shell
python opencood/utils/setup.py build_ext --inplace
```


### Train your model
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:
```python
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER} --half]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/point_pillar_fax.yaml`, meaning you want to train
CoBEVT with pointpillar backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
- `half` (optional): If set, the model will be trained with half precision. It cannot be set with multi-gpu training togetger.
- **S2R-ViT:** training the S2R-UViT model: set the yaml file `opencood\hypes_yaml\point_pillar_S2Rformer.yaml`

To train on **domain adaptation manner**, run the following command:
```python
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_da.py --hypes_yaml ${CONFIG_FILE}  --model_target  ${CHECKPOINT}   --model_source  ${CHECKPOINT}
```
- **S2R-ViT:**  the path of the training configuration file, you can set the `DA_trainin` and `da_training` in the `train_params` of yaml file in the `opencood\hypes_yaml\point_pillar_S2Rformer.yaml`;
- **S2R-ViT:** `model_target`:  the path to target domain model checkpoint (our final testing model for real data), at the beginning, you can set here as the checkpoint of the source data trained model like `model_source`;   
- **S2R-ViT:** `model_source`  the path to source domain model checkpoint (trained on simulation data (OPV2V));   


**Note that: `opencood/tools/train_da.py ` is only for Adversarial Training of Domain Adaptation;  S2R-UViT model can be directly used in the 3D object detection task**


### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `v2v4real/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```



## Citation
```shell
@inproceedings{li2024s2r,
  title={S2r-vit for multi-agent cooperative perception: Bridging the gap from simulation to reality},
  author={Li, Jinlong and Xu, Runsheng and Liu, Xinyu and Li, Baolu and Zou, Qin and Ma, Jiaqi and Yu, Hongkai},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={16374--16380},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgment
The codebase is build upon [OPV2V](https://github.com/DerrickXuNu/OpenCOOD) and [V2V4Real](https://github.com/ucla-mobility/V2V4Real), which is the first Open Cooperative Detection framework for autonomous driving.