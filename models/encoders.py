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
    def __init__(self, num_channels=3, ndf=64, input_size=32, dropout=0.1):
        super(LocalEncoder, self).__init__()
        self.input_size = input_size
        stride = 2
        padding = 1
        if input_size == 64 or input_size==32:
            kernel_size=4

        elif input_size == 28:
            kernel_size=4
            padding=1

        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(num_channels, ndf, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(ndf),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(inplace=True),
            # state size. 64 x 29 x 29, or 16 x 16

            nn.Conv2d(ndf, ndf * 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(ndf * 2),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(inplace=True))
            # state size. 128 x 26 x 26 or 8 x 8)

        if self.input_size == 64:
            self.main.add_module("Conv3", nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, \
                    stride=stride, padding=padding, bias=False))

        if self.input_size == 64 or self.input_size == 32:
            self.features_shape = [ndf * 2, 8, 8]
            self.output_shape = [ndf * 4, 4, 4]

        elif self.input_size == 28:
            self.features_shape = [ndf * 2, 7, 7]
            self.output_shape = [ndf * 4, 3, 3]

        self.output_layer = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(ndf * 4 ),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(inplace=True))

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

    def __init__(self, ndf=64, num_channels=3, input_size=32, output_size=64):
        super(GlobalEncoder, self).__init__()
        self.local_encoder = LocalEncoder(ndf=ndf, num_channels=num_channels, input_size=input_size)
        self.output_size = output_size
        self.fc_net = nn.Sequential(nn.Linear(self.local_encoder.output_size, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, self.output_size))

    def forward(self, input):
        # C: second to last conv layer, output: last conv layer
        C, output = self.local_encoder(input)
        #print(C.shape, output.shape)
        enc_input = torch.nn.Flatten()(output)
        FC = self.fc_net._modules["0"](enc_input)
        E = self.fc_net(enc_input)
        return C, FC, E.squeeze()


class ConvEncoder(nn.Module):
    def __init__(self, num_channels=3, ndf=64, code_dim=64, input_size=64, padding=1, dropout=0.1):
        super(ConvEncoder, self).__init__()
        self.code_dim = code_dim
        stride=2
        self.input_size = input_size
        self.conv_net = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(num_channels, ndf, kernel_size=3, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ndf),
            # nn.Dropout2d(p=dropout),
            nn.ReLU(inplace=True),
            # state size. 64 x 29 x 29, or 16 x 16

            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ndf * 2),
            # nn.Dropout2d(p=dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ndf * 4),
            # nn.Dropout2d(p=dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ndf * 4),
            # nn.Dropout2d(p=dropout),
            nn.ReLU(inplace=True),
        )

        if self.input_size == 64:
            self.output_shape = [ndf * 4, 4, 4]

        elif self.input_size == 28:
            self.output_shape = [ndf * 4, 2, 2]

        self.output_size = self.output_shape[0] * self.output_shape[1] * self.output_shape[2]
        self.mean = nn.Sequential(nn.Linear(self.output_size, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, self.code_dim))

        self.variance = nn.Sequential(nn.Linear(self.output_size, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, self.code_dim),
                                    nn.Softplus())

    def forward(self, input):
        # C: second to last conv layer, output: last conv layer
        C = self.conv_net(input)
        enc_input = torch.nn.Flatten()(C)
        mean = self.mean(enc_input)
        variance = self.variance(enc_input)
        I = torch.eye(self.code_dim).to(input.device)
        cov = torch.einsum('ij,ki->kij', I, variance)
        return mean, cov


class MLPEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, code_dim, prior="diagonal"):
        """
        :param input_dim:
        :param hidden_dim:
        :param code_dim:
        :param prior:
        """
        super(MLPEncoder, self).__init__()
        self.code_dim = code_dim
        self.mean = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, code_dim))

        self.cov = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, code_dim),
                                 nn.Softplus())

    def forward(self, X):
        batch_size = X.shape[0]
        mean = self.mean(X).unsqueeze(-1)
        sigma = self.cov(X)
        # make diagonal matrix from sigma
        I = torch.eye(self.code_dim).to(X.device)
        cov = torch.einsum('ij,ki->kij', I, sigma)
        return mean, cov


class IAFEncoder(nn.Module):

    def __init__(self, base_encoder):

        self.base_encoder = base_encoder

# netD = GlobalEncoder(input_size=28, num_channels=1)
# netD.apply(weights_init)
# X = torch.randn((132, 1, 28, 28))
# netD(X)
