#!/usr/bin/env bash
PYTHONPATH='./':$PYTHONPATH python train.py \
    --lr 0.05 \
    --lr_decay 0.5\
    --lr_changes 5 \
    --num_epochs 100 \
    --batch_size 10

# # Configuration for achieving: Precision: 47.6 Recall: 50.0 F1: 46.9
# # Note: training was killed at epoch 51
# PYTHONPATH='./':$PYTHONPATH python train.py \
#     --lr 0.05 \
#     --lr_decay 0.5\
#     --lr_changes 7 \
#     --num_epochs 100 \
#     --batch_size 4

# # Configuration for achieving: Precision: 47.5 Recall: 46.4 F1: 45.4
# # Note: training was killed at epoch 18
# PYTHONPATH='./':$PYTHONPATH python train.py \
#     --lr 0.05 \
#     --lr_decay 0.5\
#     --lr_changes 5 \
#     --num_epochs 60 \
#     --batch_size 10

# # Configuration for achieving: Precision: 46.5 Recall: 46.0 F1: 44.9
# # Note: Compare to the one above, this is is probably overfitting
# PYTHONPATH='./':$PYTHONPATH python train.py \
#     --lr 0.05 \
#     --lr_decay 0.5\
#     --lr_changes 5 \
#     --num_epochs 100 \
#     --batch_size 10
