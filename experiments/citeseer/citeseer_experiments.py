import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from strnn.models.strNN import StrNN


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_split_masks(data):
    """
    Visualize the masks for train, val, and test sets
    """
    pass


def load_data(
    dataset_name='CiteSeer', 
    transform=NormalizeFeatures(),
    split='full'
):
    """
    Args:
        split (str): determines the train/val/test masks for dataset
            public: public fixed split from the “Revisiting Semi-Supervised 
                Learning with Graph Embeddings” paper 
            full: all nodes except those in the validation and test sets will 
                be used for training (as in the “FastGCN: Fast Learning with 
                Graph Convolutional Networks via Importance Sampling” paper
            geom-gcn: the 10 public fixed splits from the “Geom-GCN: Geometric 
                Graph Convolutional Networks” paper    
    """
    assert split in ['public', 'full', 'geom-gcn'], f"Invalid split type: {split}"
    dataset = Planetoid(
        root='/tmp/' + dataset_name, 
        name=dataset_name, 
        transform=transform,
        split=split
    )
    data = dataset[0].to(device)
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Build adjacency matrix
    

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    load_data(split='geom-gcn')