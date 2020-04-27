#!/bin/bash
-set x
mode=fc
stride=2
dir=experiments
classifier_ckpt_1=${dir}/classifier_${mode}_local_infomax_encoder_jsd_new_prior/classifier_local_infomax_encoder_jsd_new_prior_checkpoint.pth
classifier_ckpt_3=${dir}/classifier_${mode}_local_infomax_encoder_nce_05_checkpoint.pth
classifier_ckpt_4=${dir}/classifier_${mode}_local_infomax_encoder_dv_05_checkpoint.pth
prefix1=classifier_${mode}_local_infomax_encoder_jsd_new_prior
prefix2=classifier_${mode}_local_infomax_encoder_nce_05
prefix3=classifier_${mode}_local_infomax_encoder_dv_05

for prefix in ${prefix1} ${prefix2} ${prefix3}; do
 ckpt=${dir}/${prefix}/${prefix}_checkpoint.pth
 bash slurm_scripts/launch_slurm_job.sh p100 classifier_stats_${prefix} 1 \
        "python stability_test.py --encoder_stride ${stride} --classifier_ckpt=$ckpt \
        --input_layer ${mode} --gpu --prefix=${prefix} --hidden_units=200 --batch_size=64"
done

