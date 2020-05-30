import torch
import torch.nn as nn
from utils.matrix_utils import *


class VAE(nn.Module):

    def __init__(self, encoder, decoder, prior="normal", distortion_fn=None):
        
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.code_dim = encoder.code_dim

        if prior == "normal":
            self.kl = lambda mean, cov: 0.5 * (-log_diagonal_det(cov) - cov.shape[1] + trace(cov) + torch.norm(mean, p=2, dim=1)).mean()
        # TODO: fancier priors
        self.kl = lambda mean, cov: 0.5 * (-log_diagonal_det(cov) - cov.shape[1] + trace(cov) + torch.norm(mean, p=2, dim=1)).mean()
        if distortion_fn is None:
            self.distortion = lambda X, X_hat: torch.pow(X_hat - X, 2).sum() / X.shape[0]

        else:
            self.distortion = distortion_fn

    def forward(self, X, num_samples=1):
        """

        :param X:
        :param num_samples:
        :return:
        """
        batch_size = X.shape[0]
        rank = len(X.shape) - 1
        X = X.repeat(num_samples, *([1] * rank))
        mu, cov = self.encoder(X)
        Z = torch.bmm(cov, torch.randn(batch_size * num_samples, mu.shape[1], 1).to(X.device))
        Z = Z.squeeze() + mu.squeeze()
        X_hat = self.decoder(Z)

        rate = self.kl(mu, cov)
        distortion = self.distortion(X, X_hat)
        return rate, distortion, Z, X_hat
