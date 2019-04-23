#!/usr/bin/env bash
PYTHONPATH='./':$PYTHONPATH python train.py \
    --lr 1e-10 \
    --lr_decay 0.5\
    --lr_changes 3 \
    --num_epochs 100 \
    --batch_size 12