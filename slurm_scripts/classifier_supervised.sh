#!/bin/bash

bash slurm_scripts/launch_slurm_job.sh t4 classifier_supervised 1 \
    "python3 classification.py --encoder_stride 2 --fully_supervised --gpu \
    --prefix classifier_supervised_200 --weight_decay=0.0001 --hidden_units=200 --batch_size=64 --lr=0.0001 --dropout=0.1 --epochs=2000"
