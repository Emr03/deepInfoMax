#!/bin/bash

# loop over encoder_ckpts
for dir in experiments/local_infomax_encoder_*; do
  encoder_ckpt=${dir}_checkpoint.pth
  echo ${dir}
  for mode in "fc" "conv" "y"; do
    stride=${dir:${#dir} - 1}
    dir=${dir/experiments\//}
    echo ${dir}
    # partition, j_name, resource cmd
    bash slurm_scripts/launch_slurm_job.sh p100  classifier_${mode}_${dir} 1 \
    "python3 classification.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu \
                  --prefix classifier_${mode}_${dir}"
  done
done

