import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(im_height * im_width * 3, 128)
        self.layer2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
