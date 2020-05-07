import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *
from utils.mi_estimators import *


def random_permute(X):

    '''Randomly permuts a tensor along the batch axis independently for each feature.
    Args:
        X: Input tensor.
    Returns:
        torch.Tensor: shuffled tensor.
    '''
    # random floats in X-like tensor
    b = torch.rand(X.size()).cuda()
    idx = b.sort(0)[1] # sort b values along columns, choose indices (this is a way to generate random permutations)
    adx = torch.arange(0, X.size(1)).long()
    return X[idx, adx[None, :]] # shuffle batch elements of X


class NeuralDependencyMeasure(nn.Module):

    def __init__(self, encoder, encoder_dim=64):
        super(NeuralDependencyMeasure, self).__init__()
        self.encoder = encoder
        self.encoder_dim = 64
        self.model = nn.Sequential(nn.Linear(encoder_dim, 1000),
                                   nn.ReLU(),
                                   nn.Linear(1000, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 1),
                                   )

        self.encoder.eval()

    def forward(self, X):

        with torch.no_grad():
            C, E = self.encoder(X)
        
        E = nn.Sigmoid()(E)
        # shuffle encoder units to break correlations between units
        shuffled_E = random_permute(E)
        shuffled_logits = self.model(shuffled_E)
        encoder_logits = self.model(E)
        loss = -torch.log(nn.Sigmoid()(encoder_logits) + 1E-04).mean() - \
               torch.log(torch.ones_like(shuffled_logits) - nn.Sigmoid()(shuffled_logits) + 1E-04).mean()

        # compute NWJ or DV estimate of KL
        first_term = encoder_logits.mean() # scores for joint distribution
        #second_term = torch.exp(shuffled_logits - 1.).mean()
        second_term = torch.logsumexp(shuffled_logits, dim=0).mean() - torch.log(torch.tensor(shuffled_logits.size(0)*1.)) # scores for marginal
        kl = first_term - second_term
        return loss, kl





