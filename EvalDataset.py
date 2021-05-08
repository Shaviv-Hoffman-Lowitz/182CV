import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class EvalDataset(Dataset):
    def __init__(self, main_dir, transform, classes, annot=None):
        self.main_dir = main_dir
        self.transform = transform
        self.targ = annot

        self.img_dir = os.path.join(main_dir, 'images')
        self.total_imgs = os.listdir(self.img_dir)

        if self.targ:
            self.targ = {}
            with open(os.path.join(main_dir, annot), 'r') as f:
                lines = f.read().split('\n')
                for line in lines[:-1]:
                    line = line.split('\t')
                    self.targ[line[0]] = classes.index(line[1])

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        if self.targ:
            return tensor_image, torch.tensor(self.targ[self.total_imgs[idx]])
        return tensor_image
