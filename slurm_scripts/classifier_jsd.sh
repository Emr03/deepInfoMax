#!/bin/bash

mode=fc
stride=2
dir=experiments/local_infomax_encoder_jsd_new
encoder_ckpt=${dir}/local_infomax_encoder_jsd_new_checkpoint.pth
bash slurm_scripts/launch_slurm_job.sh gpu classifier_${mode}_jsd_prior 1 \
    "python3 classification.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu \
                  --prefix classifier_${mode}_jsd_2_new --epochs 2000 --lr=0.00005 --dropout=0.1 --batch_size=64"
