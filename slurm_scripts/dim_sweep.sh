#!/bin/bash

for est in "js" "dv" 'nwj' "infonce" "smile"; do
    # partition, j_name, resource cmd
    bash slurm_scripts/launch_slurm_job.sh p100 local_infomax_encoder_${est}_2_full 2 \
    "python3 deep_infomax.py --encoder_stride 2 --mi_estimator ${est} --gpu \
                  --prefix local_infomax_encoder_${est}_2 --epochs=500 --batch_size=64"
  done
done
