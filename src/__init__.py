import os
import copy

G_LOCAL = 9.81263
G_STANDARD = 9.80665

import torch
torch.set_printoptions(precision=4)
FT = torch.DoubleTensor
torch.set_default_tensor_type(torch.DoubleTensor)

import inspect
from typing import Any, Callable, Dict, Iterable, List, Set, Union, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt

params = {'font.size': 24, 'axes.labelsize': 14,'axes.titlesize':18, 'legend.fontsize': 8, 'xtick.labelsize':12, 'ytick.labelsize':12}
plt.rcParams.update(params)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'times new roman'
# for figures with latex text use mathtext
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.default'] = 'rm'
plt.rc('legend',fontsize=12) # using a size in points

import opt_einsum as oe
def ein(expr: str, *vals):
    # return oe.contract(expr, *vals, backend='torch', optimize='greedy', use_blas=False)
    return torch.einsum(expr, *vals)
