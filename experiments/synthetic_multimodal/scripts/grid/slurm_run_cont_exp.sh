#!/bin/bash

#SBATCH -N 1
#SBATCH -p t4v2
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --output=./logs/out_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=m4
#SBATCH --time=2:00:00

source activate strnn

cd ../..
python run_experiment_mm.py --dataset_name multimodal --model_config $1 \
    --wandb_name strnn_small_cnf --flow_steps $2 --hidden_width $3 \
    --hidden_depth $4 --lr $5 --n_samples 1000 --model_seed 2541 \
    --adj_mod '["main_diagonal"]'
