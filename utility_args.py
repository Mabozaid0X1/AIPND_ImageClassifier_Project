# Imports python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image, ImageFile

import torch
from torchvision import transforms, datasets
import seaborn as sns
sns.set_style('white')

__version__ = "1.0.0"
__author__ = "Mohamed Magdy"

def train_args():
    '''
    it has 16 command line arguments by argprase module to make the user has more control
    with arguments in the model training process and if there are missing arguments,
    then the model will work with the default values.

    Command Line Arguments:
    data_dir: Image Folder 
    -- version: the version of this model
    --epochs(int): number of epochs, Default value 10
    --max_image_size(int): Image size as input size for the model
    --batch_size(int): Batch size for data loaders
    --hidden units(int): A list of hidden units in the model classifier block
    --early_stopping(int): After this number of epochs, it will stop if the valid loss is not decreasing anymore, Default None
    --lr(float): (Learning rate) a number between 0 and 1, Default value 0.01
    --dropout(float): A number between 0 and 1 as Drop out probability, Default value 0.5
    --arch(str): Pretrained CNN Model Architecture, Default value 'vgg'
    --save_dir(str): Directory to save model's checkpoint, Default value '.' (using the same directory)
    --save_summary(str): A File path to save the model training summary to csv file in it, Default None
    --checkpoint(str): A File path to Model's checkpoint that used for retraining, Default None
    --gpu: A boolean to switch between CPU and GPU, Default False
    --plot_summary: A boolean to plot the model training summary or not, Default False
    --evaluate_only: A booean to set the model for evaluation only and prevent retraining, Default False.

    This function returns these arguments as an ArgumentParser object.
    Returns:
        parse_args() --> Data structure that stores the command line arguments object  
    '''
    # Creating Parse by ArgumentParser
    
    parser = argparse.ArgumentParser(
        description='Trains a new network on a dataset and save the model as a checkpoint.'
    )
    # Creating 16 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument("--version",
                        action="version",
                        version='%(prog)s ' + __version__ + ' by ' + __author__
                       )
    
    parser.add_argument("data_dir",
                        type=str,
                        help="The path or directory of data folder"
                       )

    parser.add_argument("--max_image_size",
                        type=int,
                        default=224,
                        help="Max image size as an input size for model training"
                       )
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Batch size for data loaders"
                       )

    parser.add_argument("--arch",
                        type=str,
                        default="vgg",
                        choices=['vgg', 'resnet', 'alexnet', 'densenet', 'googlenet', 'inception'],
                        help="Name of the pretrained CNN model archticture that will use in our model\n"
                             "Note: the last 2 archticture torchvision==0.7 and torch==1.6 and more"
                       )

    parser.add_argument("--save_dir",
                        dest="save_dir",
                        type=str,
                        default='.',
                        help="Directory to save model's checkpoints in it"
                       )

    parser.add_argument("--lr",
                        type=float,
                        default=0.01,
                        help="Learning rate value for the optimizer(SGD)"
                       )

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Drop out probability to reduce the overfitting"
                       )

    parser.add_argument("--hidden_units",
                        nargs='+',
                        type=int,
                        default=[512],
                        help="List of hidden units for the model's classifier block"
                       )

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs for training our model"
                       )

    parser.add_argument("--early_stopping",
                        type=int,
                        default=None,
                        help="After this number of epochs, it will stop if the valid loss is not decreasing anymore"
                       )

    parser.add_argument("--gpu",
                        action="store_true",
                        default=False,
                        help="Use GPU for training or CPU"
                       )

    parser.add_argument("--plot_summary",
                        action="store_true",
                        default=False,
                        help="Plot the model training summary"
                       )

    parser.add_argument("--evaluate_only",
                        action="store_true",
                        default=False,
                        help="Set the model for evaluation only and prevent retraining"
                       )

    parser.add_argument("--save_summary",
                        type=str,
                        default=None,
                        help="Saves the model training summary and convert it from pandas DF to csv file"
                       )

    parser.add_argument("--check_point",
                        type=str,
                        default=None,
                        help="Model's checkpoint that will use for retraining"
                       )
    # Return pasrer object containing all argumnts
    parsing = parser.parse_args()
    
    return parsing

def predict_args():
    '''
    it has 6 command line arguments by argprase module to make the user has more control
    with arguments in the model prediction process and if there are missing arguments,
    then the model will work with the default values.
    
    Command Line Arguments:
      input(str): Image path to predict 
      checkpoint(str): A file path to Model's checkpoint for retraining
      --top_k(int): Number of top classes for predictions, Default 5
      --category_names(int): A file path to category names json file, Default None
      --gpu: A boolean to switch between GPU and CPU for prediction, Default False(CPU)
      --plot_predictions: A boolean to plot the prediction image a long with top k classes or not, Default False

    This function returns these arguments as an ArgumentParser object.
    Returns:
        parse_args() --> Data structure that stores the command line arguments object 
    '''
    # Creating Parse by  ArgumentParser
    parser = argparse.ArgumentParser(
        description='Use a pretrained model checkpoint for prediction.'
    )
    # Creating 6 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument("--input",
                        type=str,
                        help="The path or directory of data folder"
                       )
    
    parser.add_argument("--check_point",
                        type=str,
                        help="Model's checkpoint that used for the prediction"
                       )
    
    parser.add_argument("--top_k",
                        type=int,
                        default=5,
                        help="Number of top classes for predictions"
                       )
    
    parser.add_argument("--category_names",
                        type=str,
                        default=None,
                        help="file path to category names json file"
                       )

    parser.add_argument("--gpu",
                        action="store_true",
                        default=False,
                        help='Use GPU for prediction'
                       )

    parser.add_argument("--plot_predictions",
                        action="store_true",
                        default=False,
                        help="Plot the model prediction results"
                       )
    
    # Return pasrer object containing all argumnts
    parsing = parser.parse_args()
    
    return parsing


def get_dataloaders(data_dir, max_image_size, batch_size):
    '''
    it creates data loaders for model training, testing, validation based on build data transforms,
    then image datasets.
    params: 
        data_dir(str): images data dictionary folder
        max_image_size(int): the image size as an input size
        batch_size(int): the number of images as the batch size
    returns:
        dataloaders: the dataloaders object for train, valid and test dictionary
        image_datasets: the imagefolder objects for train, valid and test dictionary
        n_classes: the of train data classes and categories
        weights: the float tonser
    '''
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Task 1: Define your transforms for the training, validation, and testing sets
    # Applying image augmentation for training dataset
    # we make val_test_transform for testing and validation because we use for them the same transforms params
    train_transform = transforms.Compose([transforms.Resize((max_image_size, max_image_size)),
                                            transforms.RandomHorizontalFlip(p=0.25), # to flip the image horizontaly and randomly
                                            transforms.RandomRotation(30),  # to rotate the image in 30 degrees randomly
                                            transforms.RandomGrayscale(p=0.02),# convert the image to grayscale to train it faster
                                            transforms.RandomResizedCrop(max_image_size),
                                            transforms.ToTensor(), # to convert the numpy array of images to a tensor
                                            normalize]) # to apply the normalization

    val_test_transform = transforms.Compose([transforms.Resize(max_image_size + 1),
                                            transforms.CenterCrop(max_image_size), 
                                            transforms.ToTensor(),
                                            normalize])

    data_transforms = {
        "training": train_transform,
        "validation": val_test_transform,
        "testing": val_test_transform
    }

    # Task 2: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms["training"])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms["validation"])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms["testing"])

    image_datasets = {
        "training": train_data,
        "validation": valid_data,
        "testing": test_data
    }
    
  
    # Task 3: Using the image datasets and the trainforms, define the dataloaders (preparing dataloaders)
    train_loader = torch.utils.data.DataLoader(image_datasets["training"], batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(image_datasets["validation"], batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(image_datasets["testing"], batch_size=batch_size)

    dataloaders = {
        "training": train_loader,
        "validation": valid_loader,
        "testing": test_loader
    }

    n_classes = len(train_data.classes)

    # Get training classes weights
    targets = [s[1] for s in train_data.samples]
    classes_count = np.array([targets.count(i)
                              for i in range(n_classes)])
    weights = torch.FloatTensor(1/classes_count)
    # to check our data by printing out some data stats
    print(f'Number of training images data: {len(train_data)}')
    print(f'Number of validation images data: {len(valid_data)}')
    print(f'Number of testing images data: {len(test_data)}')
    print ("Number of classes: "+ str(len(image_datasets['training'].classes)) + "\n")
    print ("Classes: "+ str(image_datasets['training'].classes) + "\n")
    print (len(dataloaders)) # training, validation, testing

    return dataloaders, image_datasets, n_classes, weights


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image).convert("RGB")
    normalizer = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                      std = [0.229, 0.224, 0.225])
    
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalizer])
    pil_image = in_transforms(pil_image)

    return pil_image