# Structured Neural Networks

**Official implementation of the NeurIPS 2023 paper [Structured Neural Networks for Density Estimation and Causal Inference](https://openreview.net/forum?id=GaUt5aVc2N).**

![](media/main_figure.png)

## Introduction

We intrdouce the **Structured Neural Network (StrNN)**, a network architecture that enforces functional independence relationships between inputs and outputs via weight masking.

## Citation

Please use the following citation if you use this code or methods in your own work.

```bibtex
@inproceedings{
    chen2023structured,
    title = {Structured Neural Networks for Density Estimation and Causal Inference},
    author = {Asic Q Chen, Ruian Shi, Xiang Gao, Ricardo Baptista, Rahul G Krishnan},
    booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
    year = {2023},
    url = {https://openreview.net/forum?id=GaUt5aVc2N}
}
```

## Setup

To use StrNN in your project, clone this repository and run `pip install -e .` from the project root to install any required dependencies. 

## Examples

Scripts to reproduce all experiments from the paper can be found in the /experiments folder. Here we provide some code examples on how StrNN can be used.

### Binary Density Estimation with StrNN

### Real-valued Density Estimation with StrAF

### Causal Inference with StrAF
