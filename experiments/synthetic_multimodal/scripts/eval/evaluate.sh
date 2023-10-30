#!/bin/bash

#SBATCH -N 1
#SBATCH -p t4v2
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --output=./logs/out_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=m3
#SBATCH --time=4:00:00

source activate strnn

cd ../..
python evaluate_model_mm.py --wandb_name strnn_ci --model_config $1
