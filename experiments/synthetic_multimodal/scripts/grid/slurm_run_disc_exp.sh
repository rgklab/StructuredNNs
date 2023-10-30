#!/bin/bash

#SBATCH -N 1
#SBATCH -p t4v2
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --output=./logs/out_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=m5
#SBATCH --time=1:00:00

source activate strnn

cd ../..
python run_experiment_mm.py --dataset_name multimodal --model_config $1 \
    --wandb_name strnn_small --flow_steps $2 --hidden_width $3 \
    --hidden_depth $4 --lr $5 --umnn_hidden_width $6 \
    --umnn_hidden_depth $7 --n_param_per_var $8 --scheduler $9 
    --model_seed 2541 --n_samples 1000
