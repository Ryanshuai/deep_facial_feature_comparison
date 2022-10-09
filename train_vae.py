import torch
from torch import nn
import numpy as np
import os

from train_vlisulization_visdom import Visualizer
from dataloader import CelebA
from vae import VAE


def train(model: VAE, dataset, viz):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    model.train()
    for epoch in range(100):
        losses = []
        for i, (x, _) in enumerate(dataloader):
            x = x.cuda()
            model.zero_grad()
            y, mu, log_var = model(x)
            loss = mse(y, x) + 0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - 1 - log_var)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            # if viz.every_n_step(100):
            #     loss_mean = np.mean(losses)
            #     print(f"loss: {loss_mean}")
            #     viz.plot("loss", loss_mean)
            #     losses = []
            #
            # if viz.every_n_step(100):
            #     viz.cat_batch_images("im, reconstruction", (x, y), nrow=1)
            #
            # viz.tic()
        # torch.save(model.state_dict(), "pickles" + os.sep + f"_epoch_{epoch}_{round(loss_mean, 7)}" + ".pickle")


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE().to(DEVICE)
    # viz = Visualizer("vae", x_axis="step")
    viz = 0
    dataset = CelebA("E:\celebA\img_align_celeba_png")

    train(vae, dataset, viz=viz)
