import torch
import os
from tqdm import tqdm

from dataloader import CelebA
from vae_net import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE().to(DEVICE)
load_pickle = "vae_10__epoch_16_0.008683.pickle"
if os.path.exists("pickles/" + load_pickle):
    with open("pickles/" + load_pickle, "rb") as f:
        state_dict = torch.load(f, map_location=DEVICE)
    vae.load_state_dict(state_dict)

dataset = CelebA("D:\celebA\img_align_celeba_png")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

with open("vae_mse_record.txt", "w") as f:
    for i, (x, label, im_name) in tqdm(enumerate(dataloader)):
        x = x.cuda()
        with torch.no_grad():
            y, mu, log_var = vae(x, use_reparameterize=False)

        mse = torch.mean((x - y) ** 2)
        f.write(f"{mse}\n")
