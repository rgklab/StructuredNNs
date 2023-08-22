cd ..
python run_experiment.py --dataset_name multimodal --model_config ffjord_weilbach --patience 5 --wandb_name strnn_mm
python run_experiment.py --dataset_name multimodal --model_config ffjord_baseline --patience 5 --wandb_name strnn_mm
python run_experiment.py --dataset_name multimodal --model_config ffjord_strode --patience 5 --wandb_name strnn_mm
