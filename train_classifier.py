import torch
from torch import nn
import numpy as np
import os
from tqdm import tqdm

from train_vlisulization_visdom import Visualizer
from dataloader import CelebA
from classifier import Classifier


def train(model: Classifier, dataset, viz: Visualizer, save_name):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(100):
        losses = []
        kld_losses = []
        mse_losses = []
        for i, (x, label, _) in tqdm(enumerate(dataloader)):
            label = label.cuda().unsqueeze(1).float()
            x = x.cuda()
            model.zero_grad()
            y, mu, log_var = model(x)

            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
            mse_loss = bce(y, label)

            kld_weight = 5e-6
            loss = mse_loss + kld_weight * kld_loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            kld_losses.append(kld_loss.item())
            mse_losses.append(mse_loss.item())
            if viz.every_n_step(100):
                loss_mean = np.mean(losses)
                kld_loss_mean = np.mean(kld_losses)
                mse_loss_mean = np.mean(mse_losses)
                print(f"loss: {loss_mean} kld_loss: {kld_loss_mean} mse_loss: {mse_loss_mean}")
                viz.plot("loss", loss_mean)
                viz.plot("kld_loss", kld_loss_mean)
                viz.plot("mse_loss", mse_loss_mean)
                losses = []
                kld_losses = []
                mse_losses = []

            if viz.every_n_step(100):
                for i in range(8):
                    viz.imShow(f"l:{label[i].item()}, y:{round(y[i].item(), 2)}", x[i])

            viz.tic()
        torch.save(model.state_dict(),
                   "pickles" + os.sep + f"{save_name}_epoch_{epoch}_{round(loss_mean, 7)}" + ".pickle")


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = Classifier().to(DEVICE)
    load_pickle = "classifier_1_epoch_0_0.0731832.pickle"
    if os.path.exists("pickles/" + load_pickle):
        with open("pickles/" + load_pickle, "rb") as f:
            state_dict = torch.load(f, map_location=DEVICE)
        classifier.load_state_dict(state_dict)

    save_name = "classifier_2"
    viz = Visualizer(save_name, x_axis="step")
    dataset = CelebA("D:\celebA\img_align_celeba_png")

    train(classifier, dataset, viz=viz, save_name=save_name)
