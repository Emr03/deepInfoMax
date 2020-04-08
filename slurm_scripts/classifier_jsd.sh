#!/bin/bash

mode=y
stride=2
dir=experiments/local_infomax_encoder_jsd_2_full
encoder_ckpt=${dir}/local_infomax_encoder_jsd_2_full_checkpoint.pth
bash slurm_scripts/launch_slurm_job.sh p100  classifier_${mode}_jsd_2 1 \
    "python3 classification.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu \
                  --prefix classifier_${mode}_jsd_2"
