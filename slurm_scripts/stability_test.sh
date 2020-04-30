#!/bin/bash
mode=y
stride=2
dir=experiments
prefix1=classifier_${mode}_jsd_prior_200
prefix2=classifier_${mode}_jsd_no_prior_200
#prefix3=classifier_${mode}_nce_10_200
#prefix4=classifier_${mode}_local_infomax_encoder_nce_no_prior

for prefix in ${prefix1} ${prefix2}; do
 ckpt=${dir}/${prefix}/${prefix}_checkpoint.pth
 bash slurm_scripts/launch_slurm_job.sh gpu classifier_stats_${prefix} 1 \
        "python stability_test.py --encoder_stride ${stride} --classifier_ckpt=$ckpt \
        --input_layer ${mode} --gpu --prefix=${prefix} --hidden_units=200 --batch_size=64"
done

