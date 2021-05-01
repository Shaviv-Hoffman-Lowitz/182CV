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

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SpatialTransformation, DeepFool, SquareAttack, FastGradientMethod, BasicIterativeMethod
from art.defences.trainer import AdversarialTrainer

from model import Net
from torchvision import models
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
        transforms.ToTensor(),
        # these are the standard norm vectors used for imagenet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    train_set = torchvision.datasets.ImageFolder(
        data_dir / 'train', data_transforms)

    # Note: might need to try to add lots of Cuda stuff throughout the function, because I don't have it now
    
    # Changed it so that the batch size is len(train_set) to try to get training_data and training_labels to be of the right shapes
    # so that I can pass them into the fit function later on
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                    shuffle=True, num_workers=1, pin_memory=True)
    training_data_list = []
    training_labels = []
    for idx, (inputs, targets) in enumerate(train_loader):
      if idx <= 10000:
        training_data_list.append(inputs[0])
        training_labels.append(targets[0].item())
      else:
        break

    training_labels = np.asarray(training_labels)

    training_data_list = [training_data_list[index].numpy() for index in range(len(training_data_list))]

    training_data = np.asarray(training_data_list)

    # Creating a model
    model = models.resnet50(pretrained=True)

    # model.eval()

    # Freezing the weights from the pretrained model
    for param in model.parameters():
        param.requires_grad = False

    # Creating a final fully connected layer that will be trained in the training loop
    number_of_features = model.fc.in_features
    # model.fc = nn.Linear(number_of_features, 200)

    # Could alternatively have used len(CLASS_NAMES), instead of 200, like I did below
    model.fc = nn.Linear(number_of_features, len(CLASS_NAMES))
    model.to(device)

    # We should experiment with other optimizers, and the learning rate for the optimizers, as well
    optim = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)

    # I'm assuming that the model's frozen parameters stay frozen, not 100% sure if they do
    initial_classifier = PyTorchClassifier(model = model, optimizer = optim, loss = criterion, nb_classes = len(CLASS_NAMES), input_shape = (3, im_height, im_width), device_type = 'gpu', clip_values = (0.0, 1.0))


    #adversarial_attacks = []

    #adversarial_attacks.append(SpatialTransformation(initial_classifier, 20, 1, 30, 1))
    
    #adversarial_attacks.append(DeepFool(initial_classifier))

    #adversarial_attacks.append(BasicIterativeMethod(initial_classifier))

    #adversarial_attacks.append(SquareAttack(initial_classifier))

    #adversarial_attacks.append(FastGradientMethod(initial_classifier))

    # Can experiment with the last parameter
    # adversarially_trained_model = AdversarialTrainer(initial_classifier, adversarial_attacks, 0.5)


    adversarially_trained_model = AdversarialTrainer(initial_classifier, FastGradientMethod(initial_classifier), 0.1)

    adversarially_trained_model.fit(training_data, training_labels, nb_epochs = num_epochs, batch_size = batch_size)
    #adversarially_trained_model.fit(training_data, training_labels, nb_epochs = num_epochs, batch_size = 1)

    model_predictions = adversarially_trained_model.predict(training_data)
    
    model_predictions = np.argmax(model_predictions, axis = 1)

    total_correct = np.sum(model_predictions == training_labels)

    final_accuracy = total_correct/len(training_data_list)

    print("final training accuracy is: " + str(final_accuracy))

    # I think this should work, right?
    final_classifier = adversarially_trained_model.get_classifier()

    # The get_params() method returns a dictionary that maps parameter name strings to the values of those parameters
    # There is also a 'model' variable and a 'save' function that could be useful if get_params() doesn't work how we want
    # final_classifier_parameters = final_classifier.get_params()

    # I think saving it like this should be ok? Not 100% sure
    #torch.save({
         #'net': final_classifier_parameters,
    #}, 'latest.pt')

    final_classifier.save('latest.pt')
    
    #save_checkpoint({
        #'epoch': i+1,
        #'net': model.state_dict(),
        #'acc': 1.*train_correct/train_total,
    #}, 'latest.pt')


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
