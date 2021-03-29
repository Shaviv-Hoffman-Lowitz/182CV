"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import argparse
import shutil
import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import drn as models
from torch import nn


def main(args):
    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array(
        [item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = args.B
    im_height = args.H
    im_width = args.W
    num_epochs = args.E

    data_transforms = transforms.Compose([
        transforms.Resize((im_height, im_width)),
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # these are the standard norm vectors used for imagenet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    train_set = torchvision.datasets.ImageFolder(
        data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    # Creating a model
    model = models.drn_d_105(
        pretrained=True, pool_size=(im_height//8, im_width//8))

    # Freezing the weights from the pretrained model
    for param in model.parameters():
        param.requires_grad = False

    # Creating a final fully connected layer that will be trained in the training loop
    number_of_features = model.out_dim

    # Could alternatively have used len(CLASS_NAMES), instead of 200, like I did below
    model.fc = nn.Conv2d(number_of_features, len(CLASS_NAMES), kernel_size=1,
                         stride=1, padding=0, bias=True)
    model.to(device)

    # We should experiment with other optimizers as well
    # optim = torch.optim.Adam(model.parameters())
    optim = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.m, weight_decay=args.wd)

    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(num_epochs):
        train_total, train_correct = 0, 0
        start_time = time.time()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            # print("\r", end='')
            if idx % args.freq == 0:
                print(f"""
                training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}\n
                Exe Time Per Image: {(time.time()-start_time)/((1+idx)*args.B)} s
                """)
        # torch.save({
        #     'net': model.state_dict(),
        # }, 'latest.pt')
        save_checkpoint({
            'epoch': i+1,
            'net': model.state_dict(),
            'acc': 1.*train_correct/train_total,
        }, 'latest.pt')


best_acc = 0


def save_checkpoint(metadata, filename):
    global best_acc
    torch.save(metadata, filename)
    if metadata['acc'] > best_acc:
        best_acc = metadata['acc']
        shutil.copyfile(filename, 'best.pt')


if __name__ == '__main__':
    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    parser = argparse.ArgumentParser()
    parser.add_argument("-B", help="batch size", default=32, type=int)
    parser.add_argument("-H", help="image height", default=64, type=int)
    parser.add_argument("-W", help="image width", default=64, type=int)
    parser.add_argument("-E", help="num epochs", default=10, type=int)
    parser.add_argument("-lr", help="learning rate", default=0.1, type=float)
    parser.add_argument("-m", help='momentum', default=0.9, type=float)
    parser.add_argument("-wd", help="weight decay", default=1e-4, type=float)
    parser.add_argument(
        "-freq", help="print frequency, in batches", default=10, type=int)
    args = parser.parse_args()
    main(args)
