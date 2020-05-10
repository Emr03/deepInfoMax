import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *
from utils.mi_estimators import *
from utils.mi_utils import Permute
import math

class LocalDIM(nn.Module):

    def __init__(self, global_encoder, type="js", concat=False):

        super(LocalDIM, self).__init__()
        self.global_encoder = global_encoder
        self.estimator = type

        # input_shape = num_channels of local encoder output
        self.input_shape = self.global_encoder.local_encoder.output_shape
        self.loc_input_shape = self.global_encoder.local_encoder.features_shape
        self.concat = concat

        if concat:
            # since we are concatenating the global encoder representation along the channel dimension
            input_dim = self.input_shape[0] + global_encoder.output_size

            # first layer is a 1x1 conv, acts as FC layer over channel dim
            # maintain kernel size = 1 since local encoder representations are paired with global rep separately
            self.T = nn.Sequential(nn.Conv2d(in_channels=input_dim, out_channels=512,
                                             kernel_size=1, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512,
                                             kernel_size=1, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=1,
                                             kernel_size=1, stride=1))

        else:
            # two separate encoders for local feature map and global feature vector
            self.T_glob_1 = nn.Sequential(nn.Linear(self.global_encoder.output_size, 2048),
                                          nn.ReLU(),
                                          nn.Linear(2048, 2048))

            self.T_glob_2 = nn.Sequential(nn.Linear(self.global_encoder.output_size, 2048),
                                          nn.ReLU())

            self.T_loc_1 = nn.Sequential(nn.Conv2d(in_channels=self.loc_input_shape[0], out_channels=2048,
                                                   kernel_size=1, stride=1),
                                         nn.BatchNorm2d(2048),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=2048, out_channels=2048,
                                                   kernel_size=1, stride=1))

            self.T_loc_2 = nn.Sequential(nn.Conv2d(in_channels=self.loc_input_shape[0], out_channels=2048,
                                                   kernel_size=1, stride=1),
                                         nn.ReLU())

            # for layer norm bring channel dim to last, then bring back
            self.block_ln = nn.Sequential(Permute(0, 2, 3, 1),
                                          nn.LayerNorm(2048),
                                          Permute(0, 3, 1, 2))
            
            self.glob_ln = nn.LayerNorm(2048)

            # bundle up critic parts in moduleList for easy access to trainable params
            self.T = nn.ModuleList([self.T_glob_1, self.T_glob_2, self.T_loc_1, self.T_loc_2])

    def forward_concat(self, E, C):
        # TODO: fix or discard
        # replicate and concatenate E to C
        E = E.unsqueeze(2).unsqueeze(3)
        E = E.repeat(1, 1, C.shape[2], C.shape[3])

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

        # pass C, E negative pairs through 1x1 conv layers to obtain a scalar
        T = self.T(EC).squeeze()
        del EC
        torch.cuda.empty_cache()
        return T

    def forward_dot(self, E, C):

        # has shape batch_size, 2048, 8, 8 -> 64, batch_size, 2048
        embedded_local = self.block_ln(self.T_loc_1(C) + self.T_loc_2(C)).contiguous()\
            .reshape(-1, 2048, C.shape[2] * C.shape[3]).permute(2, 0, 1).contiguous()

        # has shape batch_size, 2048
        embedded_global = self.glob_ln(self.T_glob_1(E) + self.T_glob_2(E))

        # replicate embedded_global along first dimension  -> 64, batch_size, 2048
        embedded_global = embedded_global.unsqueeze(0).repeat(embedded_local.shape[0], 1, 1)

        scores = torch.bmm(embedded_local, embedded_global.transpose(2, 1))
        return scores

    def forward(self, X, return_scores=False, E=None, estimator=None):
        """
        :param X:
        :return:
        """

        if E is not None:
            Enc = E
            C, _ = self.global_encoder(X)

        else:
            # pass X through global encoder and obtain feature map C and global representation E
            C, Enc = self.global_encoder(X)

        if self.concat:
            T = self.forward_concat(E=Enc, C=C)

        else:
            # k x k, batch_size, batch_size
            T = self.forward_dot(E=Enc, C=C)

        #print(T.shape)
        # compute and return MI lower bound based on JSD, DV infoNCE or otherwise
        if not estimator:
            estimator = self.estimator

        mi = estimate_mutual_information(estimator=estimator, scores=T, baseline_fn=None, alpha_logit=None)

        if return_scores:
            return mi, Enc, T

        return mi, Enc

class GlobalDIM(nn.Module):

    def __init__(self, global_encoder, type="jsd"):
        super(GlobalDIM, self).__init__()
        self.global_encoder = global_encoder
        input_size = self.global_encoder.local_encoder.output_size + self.global_encoder.output_size
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
        C, output = self.global_encoder.local_encoder(X)
        # flatten output and concatenate with E
        output = F.flatten(output)
        E = self.global_encoder.fc_net(output)
        EC = torch.cat([E, output], dim=1)

        # pass C, E positive pairs through linear layers to obtain a scalar
        pos_T = self.T(EC)

        # shuffle C batchwise to form negative pairs
        idx = torch.randperm(C.shape[0])
        C_neg = C[idx]
        EC_neg = torch.cat([E, C_neg], dim=1)

        # pass C, E negative pairs through linear layers to obtain a scalar
        neg_T = self.T(EC_neg)

        # compute and return MI lower bound based on JSD
        return self.mi_fn(pos_T, neg_T)
