"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import argparse
import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import Net

from torch import nn


def main(args):
    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = args.B
    im_height = args.H
    im_width = args.W
    num_epochs = args.E

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    # Create a simple model
    model = Net(len(CLASS_NAMES), im_height, im_width).cuda()
    optim = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().cuda()
    for i in range(num_epochs):
        train_total, train_correct = 0,0
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            print("\r", end='')
            print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}', end='')
        torch.save({
            'net': model.state_dict(),
        }, 'latest.pt')


if __name__ == '__main__':
    print("Using GPU:", torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", help="batch size", default=32, type=int)
    parser.add_argument("-H", help="image height", default=64, type=int)
    parser.add_argument("-W", help="image width", default=64, type=int)
    parser.add_argument("-E", help="num epochs", default=1, type=int)
    args = parser.parse_args()
    main(args)
