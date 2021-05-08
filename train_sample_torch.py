"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import sys
import argparse
import shutil
import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import os
from EvalDataset import EvalDataset

import pretrainedmodels as models
from torch import nn
from ignite.metrics import Accuracy, TopKCategoricalAccuracy  # , Precision, Recall
from common import AverageMeter

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod


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
    start_epoch = 0

    data_transforms = transforms.Compose([
        transforms.Resize((im_height, im_width)),
        transforms.ToTensor(),
        # these are the standard norm vectors used for imagenet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    train_set = torchvision.datasets.ImageFolder(
        data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    val_set = EvalDataset(
        data_dir / 'val', data_transforms, sorted([item.name for item in pathlib.Path('./data/tiny-imagenet-200/train/').glob('*')]), 'val_annotations.txt')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.B,
                                             shuffle=True, num_workers=4, pin_memory=True)

    # Creating a model
    model = models.__dict__[args.model](
        num_classes=1000, pretrained='imagenet')

    # Freezing the weights from the pretrained model
    for param in model.parameters():
        param.requires_grad = False

    # Creating a final fully connected layer that will be trained in the training loop
    number_of_features = model.last_linear.in_features
    model.last_linear = nn.Linear(number_of_features, len(CLASS_NAMES))

    # This should speed up training
    model = torch.nn.DataParallel(model).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(
            args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        model.load_state_dict(checkpoint['net'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    torch.backends.cudnn.benchmark = True

    # We should experiment with other optimizers as well
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss().to(device)
    
    fast_gradient_method_attack = None
    
    if args.ads > 0:
        # Might want to mess around with what the clip_values parameter should be, I just kind of guessed
        pytorch_classifier = PyTorchClassifier(model=model, optimizer=optim, loss=criterion, nb_classes=len(
            CLASS_NAMES), input_shape=(3, im_height, im_width), device_type='gpu', clip_values=(0.0, 1.0))

        # Initialize Fast Gradient Method attack
        fast_gradient_method_attack = FastGradientMethod(pytorch_classifier)
    

    # Scheduler reduces lr after 5 epochs without loss reduction in validation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', patience=args.patience)

    for i in range(start_epoch, num_epochs):
        for param_group in optim.param_groups:
            print('Epoch [{}] Learning rate: {}'.format(
                i, param_group['lr']))

        acc, top5, avg_loss = train(
            model, train_loader, fast_gradient_method_attack, optim, criterion)
        vacc, vtop5, vloss = validate(
            model, val_loader, fast_gradient_method_attack, criterion)

        save_checkpoint({
            'epoch': i+1,
            'net': model.state_dict(),
            'acc': acc,
            'top5': top5,
            'loss': avg_loss
        }, 'latest.pt')

        # Reduce learning rate after no validation loss decrease for args.patience epochs
        scheduler.step(vloss)


best_loss = 0


def train(model, train_loader, fast_gradient_method_attack, optim, criterion):
    model.train()
    acc = Accuracy()
    top5 = TopKCategoricalAccuracy()
    avg_loss = AverageMeter()

    start_time = time.time()
    for idx, (inputs, targets) in enumerate(train_loader):

        if args.ads > 0:
            # Converting inputs to numpy array
            numpy_inputs = inputs.numpy()
            adversarial_data = fast_gradient_method_attack.generate(
                numpy_inputs)

            # Converting the adversarial data, which is currently a numpy array (I think), back to a tensor
            inputs = torch.from_numpy(adversarial_data)

        # Load x, y
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Execute
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Update Statistics
        avg_loss.update(loss.item(), inputs.size(0))
        acc.update((outputs, targets))
        top5.update((outputs, targets))

        # Backprop and step
        loss.backward()
        optim.step()
        
        if idx % args.freq == 0:
            print(f"""
                Training {100 * idx / len(train_loader):.2f}%: Top1: {acc.compute()*100:.2f} \t Top5: {top5.compute()*100:.2f}\n
                Loss: {avg_loss.val:.4f} ~ {avg_loss.avg:.4f} \n
                Exe Time Per Image: {(time.time()-start_time)/((1+idx)*args.B)}s
                """)
            sys.stdout.flush()
    return acc.compute()*100, top5.compute()*100, avg_loss.avg


def validate(model, val_loader, fast_gradient_method_attack, criterion):
    model.eval()

    start_time = time.time()
    # with torch.no_grad():
    acc = Accuracy()
    top5 = TopKCategoricalAccuracy()
    avg_loss = AverageMeter()

    for idx, (inputs, targets) in enumerate(val_loader):
        
        if args.ads > 0:
            # Converting inputs to numpy array
            numpy_inputs = inputs.numpy()
            adversarial_data = fast_gradient_method_attack.generate(
                numpy_inputs)

            # Converting the adversarial data, which is currently a numpy array (I think), back to a tensor
            inputs = torch.from_numpy(adversarial_data)

        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Update Statistics
        avg_loss.update(loss.item(), inputs.size(0))
        acc.update((outputs, targets))
        top5.update((outputs, targets))
        if idx % args.freq == 0:
            print(f"""
            Validation {100 * idx / len(val_loader):.2f}%: Top1: {acc.compute()*100:.2f} \t Top5: {top5.compute()*100:.2f}\n
            Loss: {avg_loss.val:.4f} ~ {avg_loss.avg:.4f} \n
            Exe Time Per Image: {(time.time()-start_time)/((1+idx)*args.B)}s
            """)
            sys.stdout.flush()
    print(f"""
            Validation Final: Top1: {acc.compute()*100:.2f} \t Top5: {top5.compute()*100:.2f}\n
            Loss: {avg_loss.avg:.4f} \n
            Exe Time Per Image: {(time.time()-start_time)/((1+idx)*args.B)}s
            """)
    sys.stdout.flush()

    return acc.compute()*100, top5.compute()*100, avg_loss.avg


def save_checkpoint(metadata, filename):
    global best_loss
    torch.save(metadata, filename)
    if metadata['loss'] < best_loss:
        print(
            f"""New best model with acc {metadata['acc']} / top5 {metadata['top5']}\n
            loss red {metadata['loss'] - best_loss}""")
        best_loss = metadata['loss']
        shutil.copyfile(filename, 'best.pt')


if __name__ == '__main__':
    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", help="batch size", default=256, type=int)
    parser.add_argument("-H", help="image height", default=299, type=int)
    parser.add_argument("-W", help="image width", default=299, type=int)
    parser.add_argument("-E", help="num epochs", default=10, type=int)
    parser.add_argument("-lr", help="learning rate", default=0.001, type=float)
    parser.add_argument(
        "-freq", help="print frequency, in batches", default=10, type=int)
    parser.add_argument("-model", help="model name",
                        default="inceptionresnetv2", type=str)
    parser.add_argument(
        "-resume", help="path to target model", default='', type=str)
    parser.add_argument(
        "-ads", help="probability of adversarial input", default=0.5, type=float)
    parser.add_argument(
        "-patience", help="adaptive lr patience", default=5, type=int)
    args = parser.parse_args()
    main(args)
