#!/bin/bash

mode=fc
stride=2
hidden_units=200
prefix=local_infomax_encoder_jsd_prior_matching_new
classifier_ckpt=experiments/classifier_fc_local_infomax_encoder_jsd_prior_matching_new/classifier_fc_local_infomax_encoder_jsd_prior_matching_new_checkpoint.pth
encoder_ckpt=experiments/encoders/prior/${prefix}/${prefix}_checkpoint.pth
bash slurm_scripts/launch_slurm_job.sh gpu attack_infomax 1 \
    "python3 attack_infomax.py --encoder_stride ${stride} --classifier_ckpt=${classifier_ckpt} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu --mi_estimator dv \
    --prefix ${prefix} --hidden_units=${hidden_units} --epochs 1000 --weight_decay=0.0001 --lr=0.0001 --dropout=0.1 --batch_size=64"
