import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *
from utils.mi_estimators import *
import math
import torch.distributions as dist

class PriorMatchingDiscriminator(nn.Module):

    def __init__(self, encoder_dim=64, device="cuda"):
        """
        :param encoder_dim:
        """
        super(PriorMatchingDiscriminator, self).__init__()

        self.encoder_dim = encoder_dim
        self.model = nn.Sequential(nn.Linear(encoder_dim, 1000),
                                   nn.ReLU(),
                                   nn.Linear(1000, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 1),
                                   nn.Sigmoid())

        self.device = device

        #self.prior = dist.uniform.Uniform(low=torch.tensor(0.0).to(device), high=torch.tensor(1.0).to(device)).expand(batch_shape=torch.Size([64]))

    def forward(self, E):
        N = E.shape[0]
        enc_logits = self.model(E)
        samples_prior = torch.rand(size=(N, self.encoder_dim), device=self.device)
        prior_logits = self.model(samples_prior)
        # prior_logits = self.model(self.prior.sample_n(n=N))
        # output of discriminator represents log(density ratio) of prior / mixture, (prob(prior)/[prob(encoder) + prob(prior)]) = e^D
        # then the loss is the cross entropy
        loss = -torch.log(prior_logits + 1E-04).mean() - torch.log(torch.ones_like(enc_logits) - enc_logits + 1E-04).mean()
        return loss



