#!/bin/bash

mode=y
stride=2
hidden_units=200
dir=experiments/local_infomax_encoder_jsd_new_prior
encoder_ckpt=${dir}/local_infomax_encoder_jsd_new_prior_checkpoint.pth
bash slurm_scripts/launch_slurm_job.sh gpu attack_infomax 1 \
    "python3 attack_infomax.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu \
    --prefix classifier_${mode}_jsd_prior_${hidden_units} --hidden_units=${hidden_units} --epochs 1000 --weight_decay=0.0001 --lr=0.0001 --dropout=0.1 --batch_size=64"
