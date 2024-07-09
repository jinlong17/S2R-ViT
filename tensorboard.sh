#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Jinlong Li CSU PhD
 # @Date: 2022-07-03 18:01:32
 # @LastEditors: Jinlong Li CSU PhD
 # @LastEditTime: 2023-03-28 17:22:20
### 

# source /home/jinlong/anaconda3/etc/profile.d/conda.sh


conda activate opencood





# path="/home/jinlong/4.3D_detection/model_logs/2.baseline_1feat_KPN_10_CCNet_41_point_pillar_intermediate_fusion_2022_10_18_13"
# path="/home/jinlong/4.3D_detection/model_logs/2.baseline_feat_KPN_CCNet_41_finetuned_point_pillar_intermediate_fusion_2022_10_24_01"
# path="/home/jinlong/4.3D_detection/model_logs/3.baseline_feat_FixedKPN_CCNet_41_finetuned_point_pillar_intermediate_fusion_2022_10_25_10"
# path="/home/jinlong/4.3D_detection/model_logs/baseline_no_noise_point_pillar_intermediate_fusion_2022_10_23_17"
# path="/home/jinlong/4.3D_detection/model_logs/2.baseline_1feat_KPN_0.1_CCNet_41_point_pillar_intermediate_fusion_2022_10_18_14"
# path="/home/jinlong/4.3D_detection/model_logs/2.baseline_1feat_KPN_10_CCNet_41_point_pillar_intermediate_fusion_2022_10_18_13"
# hypes_yaml="/home/jinlong/4.3D_detection/OpenCOOD/opencood/hypes_yaml/point_pillar_intermediate_KPN_1feat.yaml"

# path="/home/jinlong/4.3D_detection/model_logs/5.original_KPN_41_finetuned_point_pillar_intermediate_fusion_2022_10_09_20"



# path="/home/jinlong/4.3D_detection/model_logs/1.baseline_KPN_training_V1_point_pillar_intermediate_fusion_2022_10_29_19"
# path="//home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_intermediate_fusion_2022_11_03_22_52_59"


# path="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/1.DA_V2Vnoise_point_pillar_intermediate_fusion_2022_11_05_00_27_40"

# path="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/1.V2Vnoise_point_pillar_intermediate_fusion_2022_11_05_00_20_58"

# path="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_fcooper_2022_11_06_18_55_23"
# path="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.DA_nonoise_corpbevtlidar_2022_11_07_12_43_34"
# path="/home/jinlong/4.3D_detection/model_logs/1.baseline_KPNMAX_training_V5_point_pillar_intermediate_fusion_2022_11_14_18"
# path="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_v2vnet_2022_11_08_01_55_01"

# path="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.DA_nonoise_corpbevtlidar_2022_11_07_12_43_34"
path="/home/jinlong/jinlong_NAS/1.model_trained_saved/DA_V2V_sim2real_early_2023/1.noise_corpbevtlidar_det_stnV0_2023_03_28_14"

# path="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/DA_v0_point_pillar_intermediate_fusion_2022_11_04_00_05_25"

cd /home/jinlong/4.3D_detection/Detection_CVPR
# run python script
######## using for oringinal point pillar training
CUDA_VISIBLE_DEVICES=0  tensorboard --logdir=$path


