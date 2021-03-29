import argparse
import sys
import pathlib
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from EvalDataset import EvalDataset
from model import Net
from torchvision import models
from torch import nn
from common import AverageMeter, accuracy


def exportCSV():
    # Loop through the CSV file and make a prediction for each line
    # Open the evaluation CSV file for writing
    with open('eval_classified.csv', 'w') as eval_output_file:
        # Open the input CSV file for reading
        for line in pathlib.Path(sys.argv[1]).open():
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',')  # Extract CSV info

            print(image_id, image_path, image_height,
                  image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]
            outputs = model(img)
            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(
                image_id, CLASSES[predicted]))


def main(args):
    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])
    im_height, im_width = 64, 64

    ckpt = torch.load('best.pt')
    print("Loading Model Epoch", ckpt['epoch'], "with train acc", ckpt['acc'])

    # Creating a model
    model = models.resnext101_32x8d(pretrained=False)
    number_of_features = model.fc.in_features
    model.fc = nn.Linear(number_of_features, len(CLASSES))

    model.load_state_dict(ckpt['net'])
    model.to(device)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((args.H, args.W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        if args.CSV:
            exportCSV()
        else:

            data_dir = pathlib.Path('./data/tiny-imagenet-200')
            eval_set = EvalDataset(
                data_dir / 'val', data_transforms, CLASSES, 'val_annotations.txt')
            eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.B,
                                                      shuffle=True, num_workers=4, pin_memory=True)

            top1 = AverageMeter()
            top5 = AverageMeter()
            for idx, (inputs, targets) in enumerate(eval_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
                if idx % args.freq == 0:
                    print(
                        f'Evaluating {100 * idx / len(eval_loader):.2f}%')
            print(
                f"Accuracy on Val Set:\n\tTop 1: {top1.avg:.2f}%\n\tTop 5: {top5.avg:.2f}%")


if __name__ == '__main__':
    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-CSV', dest='CSV', help='Store Evaluation in CSV file', action='store_true')
    parser.add_argument("-B", help="batch size", default=32, type=int)
    parser.add_argument("-H", help="image height", default=64, type=int)
    parser.add_argument("-W", help="image width", default=64, type=int)
    parser.add_argument(
        "-freq", help="print frequency, in batches", default=10, type=int)
    parser.set_defaults(CSV=False)
    args = parser.parse_args()
    main(args)
