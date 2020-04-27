#!/bin/bash

# loop over encoder_ckpts
for dir in experiments/local_infomax_encoder_*; do
  echo ${dir}
  for mode in "fc" "conv" "y"; do
    stride=2
    dir=${dir/experiments\//}
    echo ${dir}
    encoder_ckpt=experiments/${dir}/${dir}_checkpoint.pth
    # partition, j_name, resource cmd
    bash slurm_scripts/launch_slurm_job.sh p100  classifier_${mode}_${dir} 1 \
    "python3 classification.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu \
                  --prefix classifier_${mode}_${dir} --batch_size=64 --hidden_units=200 --dropout=0.1 --lr=0.0001"
  done
done

