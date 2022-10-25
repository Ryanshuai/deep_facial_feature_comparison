import torch
from torch import nn

from vae import Encoder, reparameterize


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fc1 = nn.Linear(256, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        y = self.fc1(z)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        return y, mu, log_var


if __name__ == '__main__':
    model = Classifier()
    x = torch.randn(16, 3, 192, 160)
    y, mu, log_var = model(x)
    print(x.shape, y.shape, mu.shape, log_var.shape)
