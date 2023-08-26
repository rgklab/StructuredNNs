cd ..
python run_experiment.py --dataset_name multimodal --model_config straf_base --wandb_name strnn_mm --mmd_samples 256 --max_epoch 2
python run_experiment.py --dataset_name multimodal --model_config gnf_base --wandb_name strnn_mm --mmd_samples 256 --max_epoch 2
