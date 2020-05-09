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
        #E = nn.Sigmoid()(E) # we match to prior distribution after applying the Sigmoid activation, prevents large scaling of Z
        enc_logits = self.model(E)
        samples_prior = torch.rand(size=(N, self.encoder_dim), device=self.device)
        prior_logits = self.model(samples_prior)
        # output of discriminator represents log(density ratio) of prior / mixture, (prob(prior)/[prob(encoder) + prob(prior)]) = e^D
        # then the loss is the non-saturating generator loss
        gen_loss = -torch.log(enc_logits + 1E-04).mean()
        disc_loss = -torch.log(prior_logits + 1E-04).mean() - torch.log(torch.ones_like(enc_logits) - enc_logits + 1E-04).mean()
        return disc_loss, gen_loss



