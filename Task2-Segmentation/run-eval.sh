#!/usr/bin/env bash
PYTHONPATH='./':$PYTHONPATH python model/eval.py \
	--gt_label models/eval_mask_gt.pkl \
	--pred_label models/eval_mask_pred.pkl