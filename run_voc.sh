#!/usr/bin/env bash

#SBATCH -J FSCE
#SBATCH -o log.out.%j
#SBATCH -e log.err.%j
#SBATCH --partition=gpuA100_8
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
module load /home/xinyudong/anaconda3/

EXP_NAME=$1Total_Att_Scaleboxes
SAVE_DIR=/home/xinyudong/FSCE/outputs/${EXP_NAME}
IMAGENET_PRETRAIN=/home/xinyudong/FSCE/pretrain/R-101.pkl                           # <-- change it to you path
SPLIT_ID=$2

## ------------------------------- Base Pre-train ---------------------------------- #
python tools/train_net.py --num-gpus 1 \
        --config-file /home/xinyudong/FSCE/configs/PASCAL_VOC/base-training/R101_FPN_base_training_split${SPLIT_ID}.yml \
        --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}  \
               OUTPUT_DIR ${SAVE_DIR}/base${SPLIT_ID}
#
## ------------------------------ Model Preparation -------------------------------- # 
python tools/ckpt_surgery.py \
        --src1 ${SAVE_DIR}/base${SPLIT_ID}/model_final.pth \
        --method randinit \
        --save-dir ${SAVE_DIR}/base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/base${SPLIT_ID}/model_reset_surgery.pth

# ------------------------------ Novel Fine-tuning -------------------------------- #重复跑十次
# --> 1. FSRW-like, i.e. run seed0 10 times (the FSOD results on voc in most papers)
for repeat_id in 0 1 2 3 4 5 6 7 8 9
do
    for shot in 1 2 3 5 10   # if final, 10 -> 1 2 3 5 10
    do
        for seed in 0
        do
            OUTPUT_DIR=${SAVE_DIR}/novel${SPLIT_ID}/${shot}shot/${shot}shot_seed${seed}_repeat${repeat_id}
            python tools/train_net.py --num-gpus 1 \
                      --config-file configs/PASCAL_VOC/split${SPLIT_ID}/${shot}shot.yml \
                      --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}
        done
    done
done
#

