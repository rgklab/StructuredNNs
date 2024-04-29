import torch
import wandb
from torch.utils.data import DataLoader

from binary_gaussian_mnist_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    wandb.init(project="mnist_kai_ian", entity="strnn-init")

    config = wandb.config

    dataset_name = "binary_random_sparse_d30_n2000"
    adj_mtx_name = "random_sparse_d30_adj"
    opt_type = "greedy"
    train_data, val_data, adj_mtx = load_data_and_adj_mtx(dataset_name, adj_mtx_name)
    train_dl = DataLoader(train_data, batch_size=200, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=200, shuffle=False)

    data_type = "binary"
    activation = "relu"
    input_size = train_data.shape[1]
    output_size = input_size if data_type == "binary" else 2 * input_size
    hidden_sizes = [config[f'hidden_size_multiplier_{i}'] * input_size for i in
                    range(1, config['num_hidden_layers'] + 1)]
    hidden_sizes = tuple(hidden_sizes[:config['num_hidden_layers']])

    model = StrNNDensityEstimator(
        nin=input_size,
        hidden_sizes=hidden_sizes,
        nout=output_size,
        opt_type=opt_type,
        opt_args={},
        precomputed_masks=None,
        adjacency=adj_mtx,
        activation=activation,
        data_type=data_type,
        init_type="ian_uniform"
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    train_loop(model, optimizer, train_dl, val_dl, 500, 10)


if __name__ == "__main__":
    main()
