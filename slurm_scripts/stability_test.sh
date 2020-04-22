#!/bin/bash

mode=fc
stride=2
dir=experiments/
classifier_ckpt_1=${dir}/classifier_fc_jsd_prior/classifier_fc_jsd_prior_checkpoint.pth
classifier_ckpt_2=${dir}/classifier_supervised_test/classifier_supervised_test_checkpoint.pth
prefix1=classifier_fc_jsd_prior
prefix2=classifier_supervised_test

bash slurm_scripts/launch_slurm_job.sh gpu classifier_stats_1 \
    "python3 stability_test.py --encoder_stride ${stride} --classifier_ckpt=${classifier_ckpt_1} --input_layer ${mode} --gpu \
                  --prefix ${prefix1} --batch_size=64"

bash slurm_scripts/launch_slurm_job.sh gpu classifier_stats_2 \
    "python3 stability_test.py --encoder_stride ${stride} --classifier_ckpt=${classifier_ckpt_2} --input_layer ${mode} --gpu \
                  --prefix ${prefix2} --batch_size=64"