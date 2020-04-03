#!/bin/bash
#SBATCH --job-name=example_job
#SBATCH --output=/my/output/dir/%j.out
#SBATCH --error=/my/output/dir/%j.err
#SBATCH --open-mode=append
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=22G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=normal

bash example_job.sh