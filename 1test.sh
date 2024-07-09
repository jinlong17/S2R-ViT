#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Jinlong Li CSU PhD
 # @Date: 2022-07-03 18:01:32
 # @LastEditors: Jinlong Li CSU PhD
 # @LastEditTime: 2023-03-31 14:52:00
### 

# source /home/jinlong/anaconda3/etc/profile.d/conda.sh


# conda activate v2xvit



# model="/home/jinlong/jinlong_NAS/1.model_trained_saved/DA_V2V_sim2real_early_2023/0.noise_corpbevtlidar_det_V0_2023_03_29_13"
# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.nonoise_corpbevtlidar_2022_11_06_18_53_27"
# model="/home/jinlong/jinlong_NAS/1.model_trained_saved/DA_V2V_sim2real_early_2023/0.noise_corpbevtlidar_det_GRL_V0_2023_03_31_11"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/noise_dw_att_only_2023_05_12_22"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/noise_cpvit_cwin+dw_att_2023_05_12_22"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_att_only_V2_2023_05_07_21"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/cw_att_only_4_2023_05_12_22"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/only_mean_no_vit_2023_05_12_22"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/cpvit_cwin+dw_att_mean_2023_05_26_15"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/cpvit_cwin+dw_att_V2_2023_05_30_10"








# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/attfuse"




# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CoBEVT"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/V2VAM"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2x-vit"
model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CPViT"


# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/v2vnet"




# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/f-cooper"



CUDA_VISIBLE_DEVICES=4 python3 ./opencood/tools/inference.py \
    --fusion_method intermediate \
    --model_dir $model \
    --isSim 
    # --show_sequence
    # --show_vis
    # --isSim 
    # --show_sequence \
    
    

# cd ../




# exit the virtual environment
# conda deactivate