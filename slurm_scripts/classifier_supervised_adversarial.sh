#!/bin/bash

for mode in "fc" "y"; do
 bash slurm_scripts/launch_slurm_job.sh gpu  classifier_supervised_adversarial 1 \
    "python3 classification.py --encoder_stride 2 --fully_supervised  --classifier_adversarial --gpu \
                  --hidden_units=200 --prefix classifier_supervised_adversarial_200_${mode} --num_steps=20 --input_layer=${mode}"
done
