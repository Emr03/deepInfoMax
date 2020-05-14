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
        self.pre_tconv = nn.Sequential(nn.Linear(input_size, 512*4*4),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(512*4*4))

        self.tconv_model = nn.Sequential(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
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

        for param in self.tconv_model.parameters():
            weights_init(param)

    def forward(self, x):
        z = self.pre_tconv(x)
        z = z.reshape(-1, 512, 4, 4)
        out = self.tconv_model(z)
        return out

decoder = DecoderY(input_size=64)
x = torch.randn((10, 64))
print(decoder(x).shape)

