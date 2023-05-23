#!/usr/bin/env bash
set -x
DATAPATH=YOUR_DATAPATH
LOGDIR=./logs
CUDA_VISIBLE_DEVICES=0 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist filenames/kt2015_train_first20.txt --testlist filenames/kt2015_val_first20.txt \
    --logdir $LOGDIR \
    --ckpt_start_epoch 200 --summary_freq 100 \
    --epochs 8000 \
    --batch_size 4 --test_batch_size 8 \
    --lr 0.0001 \
    --maxdisp 192
