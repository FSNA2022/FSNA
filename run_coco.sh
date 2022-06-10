#!/usr/bin/env bash

#SBATCH -J FSCE
#SBATCH -o log.out.%j
#SBATCH -e log.err.%j
#SBATCH --partition=gpuA100_8
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
module load /home/xinyudong/anaconda3/

EXPNAME=$1coco
SAVE_DIR=/home/xinyudong/FSCE/outputs/${EXPNAME}
IMAGENET_PRETRAIN=/home/xinyudong/FSCE/pretrain/R-101.pkl                             # <-- change it to you path
#IMAGENET_PRETRAIN=/home/1Tm2/zjx/FSCE_outputs/coco/self_attention/base/model_0039999.pth


# ------------------------------- Base Pre-train ---------------------------------- #
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --num-gpus 4 \
    --config-file /home/xinyudong/FSCE/configs/COCO/R-101/base_training.yml   \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
           OUTPUT_DIR ${SAVE_DIR}/base
#
#
#### ------------------------------ Model Preparation -------------------------------- #
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/ckpt_surgery.py \
        --coco \
        --src1 ${SAVE_DIR}/base/model_final.pth \
        --method randinit \
        --save-dir ${SAVE_DIR}/base
BASE_WEIGHT=${SAVE_DIR}/base/model_reset_surgery.pth


### ------------------------------ Novel Fine-tuning -------------------------------- #
# --> 1. FSRW-like, i.e. run seed0 10 times (the FSOD results on coco in most papers)
for repeat_id in 0 1 2
do
    for shot in 10
    do
        for seed in 0
        do
            OUTPUT_DIR=${SAVE_DIR}/novel/${shot}shot/${shot}shot_seed${seed}_repeat${repeat_id}
            CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 \
                      --config-file /home/xinyudong/FSCE/configs/COCO/R-101/${shot}shot_baseline.yml \
                      --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}
        done
    done
done

