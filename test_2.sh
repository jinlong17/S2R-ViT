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



# model="/home/jinlong/Desktop/model_da/without_da/attfuse"
# model="/home/jinlong/Desktop/model_da/da/attfuse"
# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/3.DA_point_pillar_v2vnet_V1_2022_11_17_22_21_58"

# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_fcooper_2022_11_06_18_55_23"
# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/point_pillar_v2vnet_2022_11_08_01_55_01"

# model="/media/jinlong/Jinlong_CSU/Work/CVPR23_RunSheng/model_da/without_da/attfuse"
# model="/home/jinlong/Desktop/model_da/without_da/CoBEVT"
# model="/media/jinlong/Jinlong_CSU/Work/CVPR23_RunSheng/model_da/without_da/v2vnet"
# model="/media/jinlong/Jinlong_CSU/Work/CVPR23_RunSheng/model_da/without_da/v2x-vit"
# model="/media/jinlong/Jinlong_CSU/Work/CVPR23_RunSheng/model_da/without_da/f-cooper"
# model="/home/jinlong/Desktop/model_da/da/attfuse
# model="/home/jinlong/Desktop/model_da/da/CoBEVT"
# model="/home/jinlong/Desktop/model_da/da/v2vnet"
# model="/home/jinlong/Desktop/model_da/da/v2x-vit"
# model="/home/jinlong/Desktop/model_da/da/f-cooper"



# model="/home/jinlong/jinlong_NAS/1.model_trained_saved/DA_V2V_sim2real_early_2023/0.noise_corpbevtlidar_det_V0_2023_03_29_13"
# model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.nonoise_corpbevtlidar_2022_11_06_18_53_27"
# model="/home/jinlong/jinlong_NAS/1.model_trained_saved/DA_V2V_sim2real_early_2023/0.noise_corpbevtlidar_det_GRL_V0_2023_03_31_11"



# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/noise_cw_att_only_4_2023_05_12_22"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/cpvit_cwin+dw_att_2023_05_12_22"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_att_only_V1_2023_05_26_15"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/noise_cpvit_cwin+dw_att_mean_2023_05_30_10"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_att_only_V3_2023_06_01_18"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_att_only_V3_select_2023_06_02_22"


# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_att_only_V3_select_mean_2023_06_05_21"




# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/point_pillar_v2vnet_2023_06_08_16"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_att_only_V4_select_2023_06_07_18"


# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_att_only_V4_select_2023_06_07_18"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_att_only_V4_select_multi_2023_06_07_18"



# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/GRL_v2vnet_2023_06_10_01"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/GRL_V2VAM_2023_06_10_00"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/GRL_COBEVT_2023_06_10_00"



# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_V2VAM_2023_06_10_02"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_v2vnet_2023_06_10_01"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_COBEVT_2023_06_10_01"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_fcooper_2023_06_10_21"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_openv2v_2023_06_10_21"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_v2xvit_2023_06_10_21"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_4_8_only_2023_06_10_01"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/dw_4_8_select_multi_2023_06_10_01"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/GRL_openv2v_2023_06_11_13"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_v2xvit_2023_06_10_21"



# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/early_fusion_2023_06_11_19"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_dw_4_8_select_multi_2023_06_11_21"




# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_dw_4_8_select_att_V1_2023_06_14_22"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V1_2023_06_15_15"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V2_2023_06_15_15"


# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V3_2023_06_27_10"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V1_2023_06_26_11"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V0_2023_06_27_10"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V1_2023_06_27_10"



# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V2_2023_06_27_10"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V1_2023_06_27_10"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V0_2023_06_27_10"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V4_2023_06_28_15"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_att_V4_2023_06_28_15"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_attV1_V1_2023_06_29_10"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_attV1_V0_2023_06_29_10"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_attV2_V0_2023_06_29_10"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_attV2_V1_2023_06_29_10"

# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/DA_CPVIT_model23/no-DA/CPViT"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/1.DA_dw_4_8_select_attV4_V0_2023_07_01_00"
# model="/home/jinlongli/2.model_saved/1.da_v2vreal_model_trained2023/AdvGRL_CPViT_2023_07_01_00"
model="/home/jinlongli/2.model_saved/2.crossdata_DA2023/v2xvit_V2X_2023_07_01_00"



CUDA_VISIBLE_DEVICES=5 python3 ./opencood/tools/inference.py \
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