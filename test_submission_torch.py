import sys
import pathlib
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
#from model import Net

import pretrainedmodels as models


def main():
    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])

    # I changed these from 64 to 299
    im_height, im_width = 299, 299

    ckpt = torch.load('best.pt')
    
    # Just deleting the 'pretrained' parameter is all that we would need to do, right?
    model = models.__dict__["inceptionresnetv2"](num_classes=1000)

    # Creating a final fully connected layer
    number_of_features = model.last_linear.in_features

    # Changed it to be CLASSES instead of CLASS_NAMES because the variable was named CLASSES in this file
    model.last_linear = nn.Linear(number_of_features, len(CLASSES))

    model.load_state_dict(ckpt['net'])

    # I don't think we need the line below, do we? They didn't have it in the provided skeleton file
    # model.to(device)

    model.eval()

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
            outputs = model(img)
            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()
