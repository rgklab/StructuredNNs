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
    --wandb_name strnn_ci --model_seed $2 --persist True \
    --lr 5e-3 --scheduler plateau --adj_mod '["main_diagonal"]'
