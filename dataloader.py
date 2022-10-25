import cv2
import os
import torch
from torch.utils.data import Dataset

from glob import glob


class CelebA(Dataset):
    def __init__(self, image_dir):
        self.paths = glob(os.path.join(image_dir, '*.png'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (160, 192))
        im = im.astype('float32') / 255
        im = im.transpose(2, 0, 1)
        torch_im = torch.from_numpy(im).float()
        name = os.path.basename(path)[:-4]
        label = int(name.endswith('m'))
        return torch_im, label, os.path.basename(path)


if __name__ == '__main__':
    image_dir = "D:\celebA\img_align_celeba_png"
    dataset = CelebA(image_dir)
    for im, label, name in dataset:
        print(name)
