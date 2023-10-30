#!/bin/bash

#SBATCH -N 1
#SBATCH -p t4v2
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --output=./logs/out_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --qos=normal

source activate strnn

cd ../..
python run_experiment_mm.py --dataset_name multimodal --model_config gnf_base \
    --wandb_name strnn_test --flow_steps 10 --hidden_width 500 --hidden_depth 4 \
    --lr 1e-3 --umnn_hidden_width 250 --umnn_hidden_depth 6 \
    --n_param_per_var 50 --scheduler plateau --model_seed 2541
