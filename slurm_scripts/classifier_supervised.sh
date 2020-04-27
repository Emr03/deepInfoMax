#!/bin/bash

for mode in "fc" "conv" "y"; do
 bash slurm_scripts/launch_slurm_job.sh t4 classifier_supervised_${mode} 1 \
     "python3 classification.py --encoder_stride 2 --fully_supervised --gpu --input_layer ${mode} \
     --prefix classifier_supervised_${mode} --weight_decay=0.0001 --hidden_units=200 --batch_size=64 \
     --lr=0.0001 --dropout=0.1 --epochs=2000"
done
