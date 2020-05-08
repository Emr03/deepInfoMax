#!/bin/bash

# loop over encoder_ckpts
for dir in experiments/encoders/local_infomax_encoder_*_no_prior; do
  echo ${dir}
  for mode in "fc" "y"; do
    stride=2
    dir=${dir/experiments\/encoders\/}
    echo ${dir}
    encoder_ckpt=experiments/encoders/${dir}/${dir}_checkpoint.pth
    prefix=classifier_${mode}_${dir}
    classifier_ckpt=experiments/${prefix}/${prefix}_checkpoint.pth
    # partition, j_name, resource cmd
    bash slurm_scripts/launch_slurm_job.sh gpu classifier_${mode}_${dir}_no_prior 1 \
    "python3 classification.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu \
                  --prefix ${prefix} --batch_size=64 --hidden_units=200 --linear --dropout=0.1 --lr=0.0001 --epochs=1000 \
		  --weight_decay=0.0001"
  done
done

