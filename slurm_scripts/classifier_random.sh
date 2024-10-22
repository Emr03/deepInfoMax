#!/bin/bash

for mode in "y"; do
 bash slurm_scripts/launch_slurm_job.sh gpu classifier_supervised_${mode}_random 1 \
     "python3 classification.py --encoder_stride 2 --fully_supervised --gpu --input_layer ${mode} \
     --prefix classifier_supervised_${mode}_200_random --hidden_units=200 --random_encoder --batch_size=128 \
     --lr=0.0001 --dropout=0.0 --epochs=1000"
done
