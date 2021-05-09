import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from faug.imagenet_c_bar.transform_finder import build_transform
from faug.imagenet_c_bar.utils.converters import PilToNumpy, NumpyToTensor
import torchvision.transforms as transforms
from random import choice



def read_corruption_csv(filename = "faug/imagenet_c_bar/imagenet_c_bar.csv"):
    with open(filename) as f:
        lines = [l.rstrip() for l in f.readlines()]
    corruptions = []
    for line in lines:
        vals = line.split(",")
        if not vals:
            continue
        corruptions.extend([(vals[0], float(v)) for v in vals[1:]])
    return corruptions

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
    

class CBar(ImageFolder):
    def __init__(self, root, size):
        super().__init__(root, transform=None)
        self.size = size
        self.corruptions = read_corruption_csv()
    
    def __getitem__(self, idx):
        name, severity = choice(self.corruptions)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            PilToNumpy(),
            build_transform(name=name, severity=severity, dataset_type='imagenet'),
            NumpyToTensor(),
            transforms.Resize((self.size, self.size)),
            # these are the standard norm vectors used for imagenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
        ])
        return super().__getitem__(idx)

class EvalCBar(EvalDataset):
    def __init__(self, root, classes, size, annot=None):
        super().__init__(root, None, classes, annot)
        self.size = size
        self.corruptions = read_corruption_csv()
    
    def __getitem__(self, idx):
        name, severity = choice(self.corruptions)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            PilToNumpy(),
            build_transform(name=name, severity=severity, dataset_type='imagenet'),
            NumpyToTensor(),
            transforms.Resize((self.size, self.size)),
            # these are the standard norm vectors used for imagenet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
        ])
        return super().__getitem__(idx)