import torch
import os
from tqdm import tqdm

from classifier import Classifier
from dataloader import CelebA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = Classifier().to(DEVICE)
load_pickle = "classifier_2_epoch_5_0.0124081.pickle"
if os.path.exists("pickles/" + load_pickle):
    with open("pickles/" + load_pickle, "rb") as f:
        state_dict = torch.load(f, map_location=DEVICE)
    classifier.load_state_dict(state_dict)

dataset = CelebA("D:\celebA\img_align_celeba_png")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

with open("classifier_feature_record.txt", "w") as f:
    for i, (x, label, im_name) in tqdm(enumerate(dataloader)):
        x = x.cuda()
        with torch.no_grad():
            y, mu, log_var = classifier(x)

        BS = x.shape[0]
        for j in range(BS):
            f.write(f"{im_name[j]}  {label[j].item()}  {y[j].item()}  {mu[j].cpu().numpy().tolist()}\n")
