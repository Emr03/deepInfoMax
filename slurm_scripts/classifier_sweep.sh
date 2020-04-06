#!/bin/bash

# loop over encoder_ckpts
cd experiments/
for dir in local_infomax_encoder_*; do
  cd dir
  encoder_ckpt = ${dir}_checkpoint.pth
  for mode in "fc" "conv" "y"; do
    stride = ${dir:${#dir} - 1}
    # partition, j_name, resource cmd
    bash slurm_scripts/launch_slurm_job.sh p100  classifier_${mode}_${dir} 4 \
    "python3 classification.py --encoder_stride ${stride} --input_layer ${mode} --gpu \
                  --prefix classifier_${mode}_${dir}"
  done
done

