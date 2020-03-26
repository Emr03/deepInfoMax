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
        self.pre_tconv = nn.Sequential(nn.Linear(input_size, 512),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(512))

        # rehape to (batch_size, 512, 1, 1)
        self.tconv_model = nn.Sequential(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(256),
                                         nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=1),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(128),
                                         nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=1),
                                         nn.Tanh()
                                         )

    def forward(self, x):
        z = self.pre_tconv(x)
        z = z.unsqueeze(-1).unsqueeze(-1)
        out = self.tconv_model(z)
        return out

decoder = DecoderY(input_size=64)
x = torch.randn((10, 64))
print(decoder(x).shape)