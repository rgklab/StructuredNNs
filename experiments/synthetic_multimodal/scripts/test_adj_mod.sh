cd ..
python run_experiment.py --dataset_name multimodal --model_config ffjord_strode --max_epoch 50 --adj_mod '["main_diagonal"]'
python run_experiment.py --dataset_name multimodal --model_config ffjord_strode --max_epoch 50 --adj_mod '["main_diagonal", "reflect"]'
