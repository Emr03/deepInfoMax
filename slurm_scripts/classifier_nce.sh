#!/bin/bash

mode=y
stride=2
hidden_units=200
dir=experiments/local_infomax_encoder_nce_10
encoder_ckpt=${dir}/local_infomax_encoder_nce_10_checkpoint.pth
bash slurm_scripts/launch_slurm_job.sh gpu classifier_${mode}_nce_10 1 \
    "python3 classification.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu \
    --prefix classifier_${mode}_nce_10_${hidden_units} --hidden_units=${hidden_units} --epochs 1500 --weight_decay=0.0001 --lr=0.0001 --dropout=0.1 --batch_size=64"
