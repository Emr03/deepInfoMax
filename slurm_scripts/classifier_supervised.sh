#!/bin/bash


bash slurm_scripts/launch_slurm_job.sh gpu classifier_supervised 1 \
    "python3 classification.py --encoder_stride 2 --fully_supervised  --gpu \
                  --prefix classifier_supervised --batch_size=64 --lr=0.001 --dropout=0.1"
