import torch
import torch.nn as nn


# custom weights initialization from DCGAN
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LocalEncoder(nn.Module):
    def __init__(self, num_channels=3, ndf=64, stride=1, input_size=32, dropout=0.1):
        super(LocalEncoder, self).__init__()
        if stride == 2:
            padding = 1
            self.main = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(num_channels, ndf, kernel_size=4, stride=stride, padding=padding, bias=False),
                #nn.BatchNorm2d(ndf),
                #nn.Dropout2d(p=dropout),
                nn.ReLU(inplace=True),
                # state size. 64 x 29 x 29, or 16 x 16

                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(ndf * 2),
                #nn.Dropout2d(p=dropout),
                nn.ReLU(inplace=True))
                # state size. 128 x 26 x 26 or 8 x 8)
            
            self.features_shape = [ndf * 2, 8, 8]
            self.output_shape = [ndf * 4 * 2, 4, 4]
            
            self.output_layer = nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 4 * 2, kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(ndf * 4 * 2),
                #nn.Dropout2d(p=dropout),
                nn.ReLU(inplace=True))
        
        else:
            padding = 0
            self.main = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(num_channels, ndf, kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(ndf),
                #nn.Dropout2d(p=dropout),
                nn.ReLU(inplace=True),
                # state size. 64 x 29 x 29,

                nn.Conv2d(ndf, ndf, kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(ndf),
                #nn.Dropout2d(p=dropout),
                nn.ReLU(inplace=True),
                # state size. 128 x 26 x 26

                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(ndf * 2),
                #nn.Dropout2d(p=dropout),
                nn.ReLU(inplace=True),
                # state size (256) x 23 x 23
                )

            
            self.output_layer = nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(ndf * 4),
                #nn.Dropout2d(p=dropout),
                nn.ReLU(inplace=True))
            
            self.features_shape = [ndf * 2, 23, 23]
            self.output_shape = [ndf * 4, 20, 20]

        self.output_size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]

    def forward(self, input):
        """
        in the paper local representations are the output of the penultimate convolution layer
        :param input:
        :return:
        """
        C = self.main(input)
        output = self.output_layer(C)
        return C, output

class GlobalEncoder(nn.Module):

    def __init__(self, ndf=64, stride=1, input_size=32):
        super(GlobalEncoder, self).__init__()
        self.local_encoder = LocalEncoder(ndf=ndf, stride=stride, input_size=input_size)
        self.output_size = 64
        self.fc_net = nn.Sequential(nn.Linear(self.local_encoder.output_size, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, self.output_size))

    def forward(self, input, intermediate=False):
        C, output = self.local_encoder(input)
        # print("C shape" , C.shape)
        enc_input = torch.nn.Flatten()(output)
        if intermediate:
            return self.fc_net._modules["0"](enc_input)

        E = self.fc_net(enc_input)
        return C, E.squeeze()

netD = GlobalEncoder(stride=2)
netD.apply(weights_init)
X = torch.randn((132, 3, 32, 32))
netD(X)
