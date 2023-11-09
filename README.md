# Structured Neural Networks

**Official implementation of the NeurIPS 2023 paper [Structured Neural Networks for Density Estimation and Causal Inference](https://arxiv.org/abs/2311.02221).**

![](media/main_figure.png)

## Introduction

We introduce the **Structured Neural Network (StrNN)**, a network architecture that enforces functional independence relationships between inputs and outputs via weight masking.

## Citation

Please use the following citation if you use this code or methods in your own work.

```bibtex
@inproceedings{
    chen2023structured,
    title = {Structured Neural Networks for Density Estimation and Causal Inference},
    author = {Asic Q Chen, Ruian Shi, Xiang Gao, Ricardo Baptista, Rahul G Krishnan},
    booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
    year = {2023},
    url = {https://arxiv.org/abs/2311.02221}
}
```

## Setup

To use StrNN in your project, clone this repository and run `pip install -e .` from the project root to install any required dependencies.

## Quick Start

The StrNN provides a drop-in replacement for a fully connected neural network, allowing it to respect prescribed functional independencies.
For example, given an adjacency matrix $A$, we can initialize an StrNN as such:

```
import numpy as np
import torch
from strnn.models.strNN import StrNN

A = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0]
])

in_dim = A.shape[0]
out_dim = A.shape[1]
hid_dim = (50, 50)

strnn = StrNN(in_dim, hid_dim, out_dim, adjacency=A)

x = torch.randn(in_dim)
y = strnn(x)
```

The StrNN can be used for density estimation, for example when integrated into normalizing flows, as we show in the experiments below.

## Examples

Scripts to reproduce all experiments from the paper can be found in the /experiments folder. Here we provide some code examples on how StrNN can be used.

### Binary Density Estimation with StrNN

### Real-valued Density Estimation with StrAF

### Causal Inference with StrAF
