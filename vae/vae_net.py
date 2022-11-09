import torch
from torch import nn


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNRelu(3, 32)
        self.conv2 = ConvBNRelu(32, 64)
        self.conv3 = ConvBNRelu(64, 128)
        self.conv4 = ConvBNRelu(128, 256)
        self.conv5 = ConvBNRelu(256, 512)
        self.fc_mu = nn.Linear(512 * 6 * 5, 256)
        self.fc_log_var = nn.Linear(512 * 6 * 5, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 512 * 6 * 5)
        self.deconv1 = UpConvBNRelu(512, 256)
        self.deconv2 = UpConvBNRelu(256, 128)
        self.deconv3 = UpConvBNRelu(128, 64)
        self.deconv4 = UpConvBNRelu(64, 32)
        self.deconv5 = UpConvBNRelu(32, 32)
        self.conv = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 6, 5)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.conv(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        y = self.decoder(z)
        return y, mu, log_var


if __name__ == '__main__':
    model = VAE()
    x = torch.randn(16, 3, 192, 160)
    y, mu, log_var = model(x)
    print(x.shape, y.shape, mu.shape, log_var.shape)
