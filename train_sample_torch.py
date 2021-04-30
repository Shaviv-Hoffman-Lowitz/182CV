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

from art.estimators.classification import PyTorchClassifier, EnsembleClassifier
from art.attacks.evasion import SpatialTransformation, DeepFool, SquareAttack, FastGradientMethod, BasicIterativeMethod

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

    # Changed it so that the batch size is len(train_set) to try to get training_data and training_labels to be of the right shapes
    # so that I can pass them into the fit function later on
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set),
                                                shuffle=True, num_workers=4, pin_memory=True)

    # Need to figure out how to speed this up
    training_data = next(iter(train_loader))[0].numpy()

    # Can be one-hot encoded or not, either way is totally fine
    # Also needs to be sped up
    training_labels = next(iter(train_loader))[1].numpy()

    # Creating a model
    model = models.resnet50(pretrained=True)

    # Other models that I experimented with
    #model = models.alexnet(pretrained=True).cuda()
    #model = models.resnet101(pretrained=True).cuda()

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

    # We should experiment with other optimizers as well
    optim = torch.optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss().to(device)

    # Not sure if we should train the initial classifier on the training data before we use the initial_classifier to generate adversarial datasets
    # Definitely should experiment with both ways if we're not sure
    # Also, I'm assuming that the model's frozen parameters stay frozen, not 100% sure if they do
    initial_classifier = PyTorchClassifier(model = model, optimizer = optim, loss = criterion, nb_classes = len(CLASS_NAMES), input_shape = (3, im_height, im_width), device_type = 'gpu')
    initial_classifier.fit(training_data, training_labels, nb_epochs = num_epochs, batch_size = batch_size)

    square_attack = SquareAttack(initial_classifier)

    spatial_transformation_attack = SpatialTransformation(initial_classifier, 20, 1, 30, 1)

    deepfool_attack = DeepFool(initial_classifier)

    basic_iterative_method_attack = BasicIterativeMethod(initial_classifier)

    fast_gradient_method_attack = FastGradientMethod(initial_classifier)

    # training_labels do not need to be passed in, it is optional - should experiment with both ways
    square_attack_adversarial_dataset = square_attack.generate(training_data)

    # Labels are not optional for this attack, its not totally clear whether training_labels are supposed be one-hot encoded or not
    deepfool_adversarial_dataset = deepfool_attack.generate(training_data, training_labels)

    # Labels are not optional for this attack, its not totally clear whether training_labels are supposed be one-hot encoded or not
    spatial_transformation_adversarial_dataset = spatial_transformation_attack.generate(training_data, training_labels)

    # training_labels do not need to be passed in, it is optional - should experiment with both ways
    fast_gradient_method_adversarial_dataset = fast_gradient_method_attack.generate(training_data)

    # training_labels do not need to be passed in, it is optional - should experiment with both ways
    basic_iterative_method_adversarial_dataset = basic_iterative_method_attack.generate(training_data)

    #Creating a list of the adversarial datasets
    adversarial_datasets = [square_attack_adversarial_dataset, deepfool_adversarial_dataset, spatial_transformation_adversarial_dataset, fast_gradient_method_adversarial_dataset, basic_iterative_method_adversarial_dataset]

    # Generating 5 pretrained models, where only the last layer is not frozen - I think this is necessary, right?
    pretrained_models = []
    for num_iterations in range(5):
        model = models.resnet50(pretrained=True)

        # Freezing the weights from the pretrained model
        for param in model.parameters():
            param.requires_grad = False

        # Creating a final fully connected layer that will be trained in the training loop
        number_of_features = model.fc.in_features

        model.fc = nn.Linear(number_of_features, len(CLASS_NAMES))
        model.to(device)

        # We should experiment with other optimizers as well
        optim = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss().to(device)

        current_classifier = PyTorchClassifier(model = model, optimizer = optim, loss = criterion, nb_classes = len(CLASS_NAMES), input_shape = (3, im_height, im_width), device_type = 'gpu')
        pretrained_models.append(current_classifier)

    # Training the 5 pretrained models, each on a different adversarial dataset
    for index in range(len(pretrained_models)):
        model = pretrained_models[index]
        adversarial_dataset = adversarial_datasets[index]

        # Labels should be the same as they were originally, right?
        model.fit(adversarial_dataset, training_labels, nb_epochs = num_epochs, batch_size = batch_size)

    # Now we need to ensemble the models together (should we also include the initial_classifier, which was trained on normal, non-adversarial data, in our ensemble?)
    pretrained_models.append(initial_classifier)

    # Could experiment with giving different weights to each classifier, but by default all the classifiers get assigned the same weight
    ensemble_model = EnsembleClassifier(classifiers = pretrained_models, channels_first = True)

    ensemble_prediction_probabilities = ensemble_model.predict(x = training_data, batch_size = batch_size, raw = False)

    ensemble_predictions = np.argmax(ensemble_prediction_probabilities, axis = 1)

    total_correct = np.sum(model_predictions == training_labels)

    # I think it is fine to use image_count as the denominator, right?
    final_accuracy = total_correct/image_count
    
    print("final training accuracy is: " + str(final_accuracy))

    # The get_params() method returns a dictionary that maps parameter name strings to the values of those parameters
    ensemble_model_parameters = ensemble_model.get_params()

    # I think saving it like this should be ok? Not 100% sure
    torch.save({
         'net': ensemble_model_parameters,
    }, 'latest.pt')
    
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
