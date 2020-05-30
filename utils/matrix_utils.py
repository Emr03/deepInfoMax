import torch
import torch.nn as nn
import numpy as np

def log_diagonal_det(M):
    d = torch.diagonal(M, dim1=1, dim2=2)
    return torch.sum(torch.log(d), dim=1)

def trace(M):
    d = torch.diagonal(M, dim1=1, dim2=2)
    return torch.sum(d, dim=1)

def make_tril(elements, d):
    """

    :param elements: [batch_size, d * (d+1) / 2]
    :param d:
    :return:
    """

    assert(elements.shape[1] == d * (d + 1) / 2)



