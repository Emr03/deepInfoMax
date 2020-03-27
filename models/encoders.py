import torch
import torch.nn as nn


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LocalEncoder(nn.Module):
    def __init__(self, num_channels=3, ndf=64, input_size=32):
        super(LocalEncoder, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(num_channels, ndf, kernel_size=4, stride=1, bias=False),
            nn.ReLU(inplace=True),
            # state size. 64 x 29 x 29
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            # state size. 128 x 26 x 26
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=True),
            # state size (256) x 23 x 23
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(inplace=True),
            # state size (512) x 20 x 20
        )

        self.output_shape = [ndf * 8, 20, 20]
        self.output_size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]

    def forward(self, input):
        output = self.main(input)
        return output

class GlobalEncoder(nn.Module):

    def __init__(self, ndf=64, input_size=32):
        super(GlobalEncoder, self).__init__()
        self.local_encoder = LocalEncoder(ndf=ndf, input_size=input_size)
        self.output_size = 64
        self.fc_net = nn.Sequential(nn.Linear(self.local_encoder.output_size, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, self.output_size))

    def forward(self, input):
        C = self.local_encoder(input)
        enc_input = torch.nn.Flatten()(C)
        E = self.fc_net(enc_input)
        return C, E.squeeze()

netD = GlobalEncoder()
netD.apply(weights_init)
print(netD)
