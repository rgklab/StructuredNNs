import os

for exp in ['straf_deep', 'straf_deep_ian_init']:
    os.system(f'python run_experiment_mm.py --dataset_name multimodal --model_config {exp} --wandb_name strnn --model_seed 2541  --lr 1e-3 --scheduler plateau')