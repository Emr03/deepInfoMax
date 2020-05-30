import torch
import torch.nn as nn
from models.encoders import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DeconvDecoder(nn.Module):

    def __init__(self, input_size, output_size=64, output_channels=3, ndf=64):
        super(DeconvDecoder, self).__init__()

        self.pre_tconv = nn.Sequential(nn.Linear(input_size, ndf * 8 * 4 * 4),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(ndf * 8 * 4 * 4))
        if output_size == 64:
            self.base_shape = [ndf * 8, 4, 4]
            self.tconv_model = nn.Sequential(nn.ConvTranspose2d(in_channels=ndf * 8, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(ndf * 4),
                                             nn.ConvTranspose2d(in_channels=ndf * 4, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(ndf * 2),
                                             nn.ConvTranspose2d(in_channels=ndf * 2, out_channels=ndf, kernel_size=4, stride=2, padding=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(ndf),
                                             nn.ConvTranspose2d(in_channels=ndf, out_channels=ndf, kernel_size=4, stride=2, padding=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(ndf),
                                             nn.Conv2d(in_channels=ndf, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.Tanh()
                                             )

        elif output_size == 32:
            self.base_shape = [ndf * 8, 4, 4]
            self.tconv_model = nn.Sequential(
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh()
                )

        elif output_size == 28:
            self.base_shape = [ndf * 8, 2, 2]
            self.pre_tconv = nn.Sequential(nn.Linear(input_size, ndf * 8 * 2 * 2),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(ndf * 8 * 2 * 2))

            self.tconv_model = nn.Sequential(nn.ConvTranspose2d(in_channels=ndf * 8, out_channels=ndf * 4,
                                                                kernel_size=3, stride=1, padding=0),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(ndf * 4),
                                             nn.ConvTranspose2d(in_channels=ndf * 4, out_channels=ndf * 2,
                                                                kernel_size=3, stride=2, padding=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(ndf * 2),
                                             nn.ConvTranspose2d(in_channels=ndf * 2, out_channels=ndf, kernel_size=4,
                                                                stride=2, padding=1),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(ndf),
                                             nn.ConvTranspose2d(in_channels=ndf, out_channels=output_channels, kernel_size=4,
                                                                stride=2, padding=1),
                                             nn.Tanh()
                                             )

    def forward(self, x):
        z = self.pre_tconv(x)
        z = z.reshape(-1, self.base_shape[0], self.base_shape[1], self.base_shape[2])
        out = self.tconv_model(z)
        return out


class MLPDecoder(nn.Module):

    def __init__(self, output_dim, hidden_dim, code_dim):
        super(MLPDecoder, self).__init__()
        self.code_dim = code_dim
        self.model = nn.Sequential(nn.Linear(code_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, output_dim))

    def forward(self, Z):
        return self.model(Z)


