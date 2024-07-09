#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Jinlong Li CSU PhD
 # @Date: 2022-07-03 18:01:32
 # @LastEditors: Jinlong Li CSU PhD
 # @LastEditTime: 2022-11-20 18:27:56
### 

# source /home/jinlong/anaconda3/etc/profile.d/conda.sh


conda activate v2xvit

model="/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/logs/2.nonoise_corpbevtlidar_2022_11_06_18_53_27"

# cd /home/jinlong/4.3D_detection/v2vreal-main
# run python script
CUDA_VISIBLE_DEVICES=1 python3 v2vreal/opencood/tools/inference.py \
    --fusion_method intermediate \
    --model_dir $model \
    --isSim
    # --show_vis
    # --isSim 
    # --show_sequence \
    
    


# cd ../




# exit the virtual environment
# conda deactivate