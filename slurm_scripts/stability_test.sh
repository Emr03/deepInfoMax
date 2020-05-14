#!/bin/bash
mode=y
stride=2

for dir in experiments/infomax_classifiers/classifier_${mode}_*; do
 prefix=${dir/experiments\/}
 name=${prefix/infomax_classifiers\/}
 ckpt=${dir}/${name}_checkpoint.pth
 bash slurm_scripts/launch_slurm_job.sh gpu classifier_stats_${name} 1 \
        "python stability_test.py --encoder_stride ${stride} --classifier_ckpt=$ckpt \
        --input_layer ${mode} --gpu --prefix=${prefix} --hidden_units=200 --batch_size=64"
done

