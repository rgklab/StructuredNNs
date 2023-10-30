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
python run_experiment_mm.py --dataset_name multimodal --model_config ffjord_baseline \
    --wandb_name strnn_test --flow_steps 5 --hidden_width 100 \
    --hidden_depth 3 --lr 1e-3 --n_samples 1000 --model_seed 2541 \
    --adj_mod '["main_diagonal"]'

