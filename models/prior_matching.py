import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *
from utils.mi_estimators import *
import math
import torch.distributions as dist

class PriorMatchingDiscriminator(nn.Module):

    def __init__(self, encoder_dim=64):
        """
        :param encoder_dim:
        """
        super(PriorMatchingDiscriminator, self).__init__()

        self.encoder_dim = encoder_dim
        self.model = nn.Sequential(nn.Linear(encoder_dim, 1000),
                                   nn.ReLU(),
                                   nn.Linear(1000, 200),
                                   nn.ReLU())

        self.prior = dist.uniform.Uniform(low=0, high=1).expand(batch_shape=torch.Size([64]))

    def forward(self, E):
        N = E.shape[0]
        enc_logits = self.model(E)
        prior_logits = self.model(self.prior.sample_n(n=N))
        # output of discriminator represents log(density ratio) of prior / mixture, (prob(prior)/[prob(encoder) + prob(prior)]) = e^D
        # then the loss is the Jensen-Shannon divergence between Prior and P(encoder)
        loss = torch.log(prior_logits).mean() + torch.log(torch.ones_like(enc_logits) - enc_logits).mean()
        return loss



