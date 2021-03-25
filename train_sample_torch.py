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
from torchvision import models
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

    # Checking if cuda is available
    cuda_available = torch.cuda.is_available()

    # Creating a model
    if cuda_available:
        model = models.resnet50(pretrained=True).cuda()
    else:
        model = models.resnet50(pretrained=True)

    # Other models that I experimented with 
    #model = models.alexnet(pretrained=True).cuda()
    #model = models.resnet101(pretrained=True).cuda()

    #model.eval()

    # Freezing the weights from the pretrained model
    for param in model.parameters():
      param.requires_grad = False

    # Creating a final fully connected layer that will be trained in the training loop
    number_of_features = model.fc.in_features
    model.fc = nn.Linear(number_of_features, 200)

    # Could alternatively have used len(CLASS_NAMES), instead of 200, like I did below
    #model.fc = nn.Linear(number_of_features, len(CLASS_NAMES))

    # We should experiment with other optimizers as well
    optim = torch.optim.Adam(model.parameters())

    if cuda_available:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    for i in range(num_epochs):
        train_total, train_correct = 0,0
        for idx, (inputs, targets) in enumerate(train_loader):
            # Sometimes it does not print training accuracies as the model is training, so uncomment the line below to see accuracies during training
            #print("printing accuracies during training")
            if cuda_available:
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
    parser.add_argument("-E", help="num epochs", default=100, type=int)
    args = parser.parse_args()
    main(args)
