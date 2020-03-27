import torch
import torch.nn as nn
from models.encoders import *

class DecoderY(nn.Module):
    """
    Decoder for reconstructing inputs
    """
    def __init__(self, input_size):
        """

        :param input_size: shape of global representation
        :param img_size:
        """
        super(DecoderY, self).__init__()
        self.pre_tconv = nn.Sequential(nn.Linear(input_size, 256),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(256))

        self.tconv_model = nn.Sequential(nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=5, stride=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(256),
                                         nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(128),
                                         nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(64),
                                         nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(32),
                                         nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(16),
                                         nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1),
                                         nn.Tanh()
                                         )

    def forward(self, x):
        z = self.pre_tconv(x)
        z = z.reshape(-1, 1, 16, 16)
        out = self.tconv_model(z)
        return out

decoder = DecoderY(input_size=64)
x = torch.randn((10, 64))
print(decoder(x).shape)

