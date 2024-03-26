# Beta test 1

import numpy as np
import torch
from strnn.models.strNN import StrNN

# import torch
print('cuda?')
print(torch.cuda.is_available())

# wandb API key
# 254cf55e0c4cbbbed6e5426bd0db54fcd97087e3

# true API key: 
# 254cf55e0c4cbbbed6e5426bd0db54fcd97087e3

# # print(A)
# A = np.array([
#     [0, 0, 0, 0],
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [1, 0, 1, 0]
# ])

# print(A)

# out_dim = A.shape[0]
# in_dim = A.shape[1]
# hid_dim = (50, 50)

# strnn = StrNN(in_dim, hid_dim, out_dim, opt_type='greedy', adjacency=A)

# print(strnn)

# x = torch.randn(in_dim)
# print(x)
# y = strnn(x)

print('--------------Half--------------')

from strnn.models.discrete_flows import AutoregressiveFlowFactory
from strnn.models.continuous_flows import ContinuousFlowFactory

A = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0]
])

af_config = {
    "input_dim": A.shape[0],
    "adjacency_matrix": A,
    "base_model": "ANF",
    "opt_type": "greedy",
    "opt_args": {},
    "conditioner_type": "strnn",
    "conditioner_hid_dim": [50, 50],
    "conditioner_act_type": "relu",
    "normalizer_type": "umnn",
    "umnn_int_solver": "CC",
    "umnn_int_step": 20,
    "umnn_int_hid_dim": [50, 50],
    "flow_steps": 10,
    "flow_permute": False,
    "n_param_per_var": 25
}

straf = AutoregressiveFlowFactory(af_config).build_flow()

x = torch.randn(1, A.shape[0])
z, jac = straf(x)
x_bar = straf.invert(z)

print('z;jac',z,'\n',jac)
print(x_bar)


# cnf_config = {...} # See model_config.yaml for values
# strcnf = ContinuousFlowFactory(cnf_config).build_flow()
# z, jac = strcnf(x)
# x_bar = strcnf.invert(z)