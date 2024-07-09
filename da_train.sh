#!/bin/sh



# model_source=""
# model_target=""







# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/V2VAM"
# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml"


# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT"
# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_fax.yaml"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2vnet"
# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_v2vnet.yaml"


# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/attfuse"
# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_opv2v.yaml"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/f-cooper"
# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_fcooper.yaml"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2x-vit"
# hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_transformer.yaml"

# model_source="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CPViT/net_epoch84.pth"
# model_target="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CPViT/net_epoch84.pth"

model_source="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CPViT/net_epoch84.pth"
# model_target="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_CPViT_2023_07_01_00/1/net_epoch20.pth"
model_target="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_CPViT_2023_07_01_00/1/net_epoch8.pth"
# model_target="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CPViT/net_epoch84.pth"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CPViT"
hypes_yaml="/home/jinlongli/1.Detection_Set/DA_V2V/opencood/hypes_yaml/point_pillar_fax_deformable.yaml"

######## using for DA point pillar training
CUDA_VISIBLE_DEVICES=3 python3 ./opencood/tools/train_da.py  --hypes_yaml $hypes_yaml --model_target $model_target   --model_source $model_source 


# CUDA_VISIBLE_DEVICES=0  python3 ./opencood/tools/train_da.py  --hypes_yaml $hypes_yaml --model $model_target   #--model_source $model_source 