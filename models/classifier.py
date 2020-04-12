import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import *
import math

def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_weights(model):
    for param in model.parameters():
        param.requires_grad = True

class ClassifierConv(nn.Module):

    def __init__(self, encoder, num_classes, hidden_units=1024, dropout=0.1, freeze_encoder=True):
        super(ClassifierConv, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.input_size = encoder.local_encoder.output_size
        self.model = nn.Sequential(nn.Flatten(),
                                   nn.Linear(self.input_size, hidden_units),
                                   nn.Dropout(p=dropout),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, num_classes))

        if freeze_encoder:
            # since input to the classifier is the
            freeze_weights(self.encoder)
            self.encoder.eval()

    def forward(self, X):
        C, E = self.encoder(X)
        return self.model(C)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

class ClassifierFC(nn.Module):

    def __init__(self, encoder, num_classes, hidden_units=1024, freeze_encoder=True, dropout=0.1):
        super(ClassifierFC, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.input_size = encoder.local_encoder.output_size
        self.layer_1 = self.encoder.fc_net._modules['0']
        self.layer_2 = nn.Linear(hidden_units, num_classes)
        self.model = nn.Sequential(self.layer_1, nn.Dropout(p=dropout), nn.ReLU(), self.layer_2)

        if freeze_encoder:
            # freeze encoder, but reset layer1 to Trainable
            freeze_weights(encoder)
            self.encoder.eval()
            unfreeze_weights(self.layer_1)

    def forward(self, X):
        C, E = self.encoder(X)
        C = nn.Flatten()(C)
        return self.model(C)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

class ClassifierY(nn.Module):

    def __init__(self, encoder, num_classes, hidden_units=1024, freeze_encoder=True, dropout=0.1):
        super(ClassifierY, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.input_size = encoder.output_size
        self.model = nn.Sequential(nn.Linear(self.input_size, hidden_units),
                                   nn.Dropout(p=dropout),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, num_classes))

        if freeze_encoder:
            # since input to classifier is output of encoder Y, freeze all of the encoder
            freeze_weights(self.encoder)
            self.encoder.eval()

    def forward(self, X):
        C, E = self.encoder(X)
        return self.model(E)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()



