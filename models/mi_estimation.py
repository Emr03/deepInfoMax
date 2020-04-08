import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *
import math

def mi_jsd(t_pos, t_neg):
    return torch.mean(F.softplus(-t_pos)) + torch.mean(F.softplus(t_neg)))

def mi_dv(t_pos, t_neg):
    return -torch.mean(t_pos) + torch.log(torch.mean(torch.exp(t_neg)))

def mi_nce(t_pos, t_neg):
    return -torch.mean(t_pos) + torch.mean(torch.log(torch.sum(torch.exp(t_neg))))

class LocalDIM(nn.Module):

    def __init__(self, global_encoder, type="jsd"):
        super(LocalDIM, self).__init__()
        self.global_encoder = global_encoder

        if type == "jsd":
            self.mi_fn = mi_jsd

        elif type == "dv":
            self.mi_fn = mi_dv

        elif type == "nce":
            self.mi_fn = mi_nce

        # TODO type == "W"

        # input_shape = num_channels of local encoder output
        input_shape = self.global_encoder.local_encoder.output_shape

        # since we are concatenating the global encoder representation along the channel dimension
        input_shape[0] = input_shape[0] + global_encoder.output_size

        # first layer is a 1x1 conv, acts as FC layer over channel dim
        # maintain kernel size = 1 since local encoder representations are paired with global rep separately
        self.T = nn.Sequential(nn.Conv2d(in_channels=input_shape[0], out_channels=512,
                                         kernel_size=1, stride=1),
                               nn.ReLU(),
                               nn.Conv2d(in_channels=512, out_channels=512,
                                         kernel_size=1, stride=1),
                               nn.ReLU(),
                               nn.Conv2d(in_channels=512, out_channels=1,
                                         kernel_size=1, stride=1))

    def forward(self, X):
        """
        :param X:
        :return:
        """
        # pass X through global encoder and obtain feature map C and global representation E
        C, E = self.global_encoder(X)

        # replicate and concatenate E to C
        E = E.unsqueeze(2).unsqueeze(3)
        E = E.repeat(1, 1, C.shape[2], C.shape[3])

        EC = torch.cat([E, C], dim=1)

        # pass C, E positive pairs through 1x1 conv layers to obtain a scalar
        pos_T = self.T(EC)

        # Each element along the batch dimension in C should be mapped with every negative element in E
        batch_size = C.shape[0]
        k = C.shape[2]
        d = E.shape[1]

        # E has shape (batch_size, dim, k, k) -> (dim, batch_size, k*k) -> (dim, batch_size * k * k)
        E = E.reshape(batch_size, -1, k * k).transpose(1, 0).reshape(-1, batch_size * k * k)
        C = C.reshape(batch_size, -1, k * k).transpose(1, 0).reshape(-1, batch_size * k * k)

        # Add dim to E and C, repeat the batch batch_size * k * k times, to obtain N * N samples
        E = E.unsqueeze(2).repeat(1, 1, batch_size * k * k)
        C = C.unsqueeze(1).repeat(1, batch_size * k * k, 1)
        N = batch_size * k * k

        # concatentate, note that EC[:, i, i] are the positive samples
        # shape = (1, num_channels + dim, N, N)
        EC = torch.cat([C, E], dim=0).unsqueeze(0)

        # diagonal entries correspond to positive samples 
        mask = 1 - torch.eye(N)
        mask = mask.unsqueeze(0).unsqueeze(1)
        
        # pass C, E negative pairs through 1x1 conv layers to obtain a scalar
        neg_T = self.T(EC)
        del EC
        torch.cuda.empty_cache()

        # compute and return MI lower bound based on JSD, DV infoNCE or otherwise
        return self.mi_fn(pos_T, neg_T)

class GlobalDIM(nn.Module):

    def __init__(self, global_encoder, type="jsd"):
        super(GlobalDIM, self).__init__()
        self.global_encoder = global_encoder
        input_size = self.global_encoder.local_encoder.output_size
        self.T = nn.Sequential(nn.Linear(input_size, 512),
                               nn.ReLU(),
                               nn.Linear(512, 512),
                               nn.ReLU(),
                               nn.Linear(512, 1))

    def forward(self, X):
        """

        :param X:
        :return:
        """
        # pass X through global encoder and obtain feature map C and global representation E
        C, E = self.global_encoder(X)

        # flatten C and concatenate with E
        C = F.flatten(C)
        EC = torch.cat([E, C], dim=1)

        # pass C, E positive pairs through linear layers to obtain a scalar
        pos_T = self.T(EC)

        # shuffle C batchwise to form negative pairs
        idx = torch.randperm(C.shape[0])
        C_neg = C[idx]
        EC_neg = torch.cat([E, C_neg], dim=1)

        # pass C, E negative pairs through linear layers to obtain a scalar
        neg_T = self.T(EC_neg)

        # compute and return MI lower bound based on JSD, or infoNCE
        return self.mi_fn(pos_T, neg_T)
