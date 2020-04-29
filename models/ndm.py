import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *
from utils.mi_estimators import *

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
                                   nn.Sigmoid())

        self.encoder.eval()

    def forward(self, X):

        with torch.no_grad:
            E = self.encoder(X)
        # shuffle encoder units to break correlations between units
        idx = torch.randperm(E.shape[-1])
        shuffled_E = E[idx]
        shuffled_logits = self.model(shuffled_E)
        encoder_logits = self.model(E)
        loss = -torch.log(encoder_logits + 1E-04).mean() - torch.log(torch.ones_like(shuffled_logits) - shuffled_logits + 1E-04).mean()
        return loss





