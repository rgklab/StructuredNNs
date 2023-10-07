#!/bin/bash

#SBATCH -N 1
#SBATCH -p t4v2,rtx6000
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --output=./logs/out_%j.txt
#SBATCH --gres=gpu:1

source activate strnn

cd ..
python run_experiment.py --dataset_name multimodal --model_config $1 \
    --wandb_name strnn_mm --flow_steps $2 --hidden_width $3 --hidden_depth $4 \
    --lr $5 --mmd_samples 512
