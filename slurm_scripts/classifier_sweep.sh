#!/bin/bash

# loop over encoder_ckpts
for dir in experiments/local_infomax_encoder_*; do
  echo ${dir}
  for mode in "fc"; do
    stride=2
    dir=${dir/experiments\//}
    echo ${dir}
    encoder_ckpt=experiments/${dir}/${dir}_checkpoint.pth
    classifier_ckpt=classifier_${mode}_${dir}_adv_200_checkpoint.pth
    # partition, j_name, resource cmd
    bash slurm_scripts/launch_slurm_job.sh gpu adv_classifier_${mode}_${dir} 1 \
    "python3 classification.py --encoder_stride ${stride} --classifier_ckpt=${classifier_ckpt} --input_layer ${mode} --gpu \
                  --prefix classifier_${mode}_${dir}_adv_200 --classifier_adversarial --batch_size=64 --hidden_units=200 --dropout=0.1 --lr=0.0001 --epochs=2000"
  done
done

