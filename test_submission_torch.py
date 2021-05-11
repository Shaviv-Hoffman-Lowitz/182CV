import sys
import pathlib
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
#from model import Net
from torch import nn

import pretrainedmodels as models


def main():
    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])

    cuda = torch.device('cuda:0')

    # I changed these from 64 to 299
    im_height, im_width = 299, 299



    art_fgm_checkpoint = torch.load('artfgm.pt')
    
    # Just deleting the 'pretrained' parameter is all that we would need to do, right?
    art_fgm_model = models.__dict__["inceptionresnetv2"](num_classes=1000, pretrained=None)

    # Creating a final fully connected layer
    number_of_features = art_fgm_model.last_linear.in_features

    # Changed it to be CLASSES instead of CLASS_NAMES because the variable was named CLASSES in this file
    art_fgm_model.last_linear = nn.Linear(number_of_features, len(CLASSES))

    art_fgm_model = torch.nn.DataParallel(art_fgm_model).to(device=cuda)

    art_fgm_model.load_state_dict(art_fgm_checkpoint['net'])

    # Do we need the line below? They didn't have it in the provided skeleton file
    # art_fgm_model.to(device=cuda)

    art_fgm_model.eval()




    robustness_checkpoint = torch.load('robustness.pt')
    
    # Just deleting the 'pretrained' parameter is all that we would need to do, right?
    robustness_model = models.__dict__["inceptionresnetv2"](num_classes=1000, pretrained=None)

    # Creating a final fully connected layer
    number_of_features = robustness_model.last_linear.in_features

    # Changed it to be CLASSES instead of CLASS_NAMES because the variable was named CLASSES in this file
    robustness_model.last_linear = nn.Linear(number_of_features, len(CLASSES))

    robustness_model = torch.nn.DataParallel(robustness_model).to(device=cuda)

    robustness_model.load_state_dict(robustness_checkpoint['net'])

    # Do we need the line below? They didn't have it in the provided skeleton file
    # robustness_model.to(device=cuda)

    robustness_model.eval()





    facebook_research_checkpoint = torch.load('facebook_research.pt')
    
    # Just deleting the 'pretrained' parameter is all that we would need to do, right?
    facebook_research_model = models.__dict__["inceptionresnetv2"](num_classes=1000, pretrained=None)

    # Creating a final fully connected layer
    number_of_features = facebook_research_model.last_linear.in_features

    # Changed it to be CLASSES instead of CLASS_NAMES because the variable was named CLASSES in this file
    facebook_research_model.last_linear = nn.Linear(number_of_features, len(CLASSES))

    facebook_research_model = torch.nn.DataParallel(facebook_research_model).to(device=cuda)

    facebook_research_model.load_state_dict(facebook_research_checkpoint['net'])

    # Do we need the line below? They didn't have it in the provided skeleton file
    # robustness_model.to(device=cuda)

    facebook_research_model.eval()






    data_transforms = transforms.Compose([
        transforms.Resize((im_height, im_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # Loop through the CSV file and make a prediction for each line
    with open('eval_classified.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for line in pathlib.Path(sys.argv[1]).open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',')  # Extract CSV info

            # They had the print statement below in their skeleton code, should we keep it in or not?
            # print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]

            # Using Cuda, we want to do this, right?
            img = img.to(device=cuda)

            art_fgm_outputs = art_fgm_model(img)

            robustness_outputs = robustness_model(img)

            facebook_research_outputs = facebook_research_model(img)


            outputs = (art_fgm_outputs + robustness_outputs + facebook_research_outputs) / 3


            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()
