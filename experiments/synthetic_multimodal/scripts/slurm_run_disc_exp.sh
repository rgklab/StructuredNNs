#!/bin/bash

#SBATCH -N 1
#SBATCH -p t4v2,rtx6000
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --output=./logs/out_%j.txt
#SBATCH --gres=gpu:4

source activate strnn

cd ..
python run_experiment.py --dataset_name multimodal --model_config $1 \
    --wandb_name strnn_grid --flow_steps $2 --hidden_width $3 \
    --hidden_depth $4 --lr $5 --mmd_samples 128 --umnn_hidden_width $6 \
    --umnn_hidden_depth $7 --n_param_per_var $8 --scheduler $9
