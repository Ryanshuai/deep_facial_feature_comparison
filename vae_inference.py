import torch
import os
import cv2
from vae import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = VAE().to(DEVICE)
load_pickle = "vae_finetune_epoch_9_0.0089038.pickle"
if os.path.exists("pickles/" + load_pickle):
    with open("pickles/" + load_pickle, "rb") as f:
        state_dict = torch.load(f, map_location=DEVICE)
    classifier.load_state_dict(state_dict)

image_path = r"D:\celebA\img_align_celeba_png\000001_f.png"
# image_path = r"D:\celebA\img_align_celeba_png\000003_m.png"


im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (160, 192))
im = im.astype('float32') / 255
im = im.transpose(2, 0, 1)
torch_im = torch.from_numpy(im).float()
torch_im = torch_im.unsqueeze(0)
torch_im = torch_im.to(DEVICE)

with torch.no_grad():
    y, mu, log_var = classifier(torch_im)

print(y, mu, log_var)
