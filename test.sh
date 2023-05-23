#!/usr/bin/env bash
set -x
DATAPATH=YOUR_DATAPATH
LOGDIR=./logs/test
MODEL=logs/trained_model/kitti.ckpt
CUDA_VISIBLE_DEVICES=0 python test.py --dataset kitti \
    --datapath $DATAPATH --testlist filenames/kt2015_val_first20.txt \
    --logdir $LOGDIR \
    --resume  $MODEL\
    --summary_freq 20 \
    --test_batch_size 8 \
    # --save_disp_to_file
