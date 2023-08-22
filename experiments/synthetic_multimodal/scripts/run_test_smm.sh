cd ..
python run_experiment.py --dataset_name multimodal --model_config ffjord_weilbach --max_epoch 1
python run_experiment.py --dataset_name multimodal --model_config ffjord_baseline --max_epoch 1
python run_experiment.py --dataset_name multimodal --model_config ffjord_strode --max_epoch 1
