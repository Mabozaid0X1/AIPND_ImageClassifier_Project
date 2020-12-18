# Imports python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import time
from datetime import datetime
from tqdm import tqdm

from utility_args import process_image
import json
import torch
from torchvision import models
from torch import optim, nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

# for training to be robust to truncated images
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Setting the pretrained CNN model architectures to use
densenet121 = models.densenet121(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

# if your torchvision 0.7=< and torch 1.6=< you can use these models
# googlenet = models.googlenet(pretrained=True)
# inception = models.inception_v3(pretrained=True, aux_logits=False)

models = {'densenet': densenet121, 'resnet': resnet18, # 'inception': inception, 'googlenet': googlenet,
          'alexnet': alexnet, 'vgg': vgg16}
class Classifier():
    def __init__(self, device):
        '''
        Starting the classifier class with the required parameters
        param:
        model(str): model name that we will use
        device: using for training, or predicting CPU or GPU (CUDA)
        '''
        self.device = device
        # creating a starting point (minimum validation loss) to decrease the valid_loss
        self.valid_loss_min = np.Inf
        # creating empty lists to monitor training and validation losses and accuracy for each epoch
        self.train_loss_list = []
        self.train_acc_list = []
    
        self.valid_loss_list = []
        self.valid_acc_list = []
        # Initialize start epoch
        self.start_epoch = 1
        # Initialize cat to name file with None
        self.cat_to_name = None

    def _creating_classifier(self, n_inputs, n_outputs, hidden_units=[512], dropout=0.5):
        """
        it creates a classifier for using in the model
        param:
        n_inputs(int): number of input features
        n_outputs(int): number of output features
        hidden_units(list): a list of integers we will use it as hidden units
        dropout(float): a number between 0 and 1 to be used as the probability for dropout(to reduce the overfitting)
        """
        classifier = nn.ModuleList()
        classifier.append(nn.Linear(n_inputs, hidden_units[0]))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(dropout))
        if len(hidden_units) > 1:
            for (h1, h2) in zip(hidden_units[:-1], hidden_units[1:]):
                classifier.append(nn.Linear(h1, h2))
                classifier.append(nn.ReLU())
                classifier.append(nn.Dropout(dropout))

        classifier.append(nn.Linear(hidden_units[-1], n_outputs))
        return nn.Sequential(*classifier)

    def compile(self, n_outputs, arch='vgg', hidden_units=[512], dropout=0.5, lr=0.01, weights=None):
        '''
        Setting the model archticture,
        number of hidden layers to use,
        numer of hidden units for each layer,
        dropout,
        optimizer,
        learning rate, and
        weights.
        to prepare the model and compile it for training.
        param:
        arch(str): the name of the pretrained CNN model archticture
        hidden_units(list): number of hidden units in a list we will use it for each hidden layer
        dropout(float): a number between 0 and 1 to be used as the probability for dropout
        lr(float): learning rate is a number between 0 and 1 to be used for optimizer learning rate
        weights(tensor): a tensor with classes weights to be used for criterion
        '''
        # Creating a model
        self.model = models[arch]
        self.arch = arch

        # to freeze the model parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # getting the input features for the model classifier layer
        if arch == 'vgg':
            n_inputs = self.model.classifier[0].in_features
        elif arch == 'alexnet':
            n_inputs = self.model.classifier[1].in_features
        elif arch == 'densenet':
            n_inputs = self.model.classifier.in_features
        elif arch in ['resnet']: # , 'inception', 'googlenet'
            n_inputs = self.model.fc.in_features
        else:
            print('ERROR 404 NOT FOUND\n'
                f'{arch} is not available\n'
                f'Can you choose another archticture from {models.keys()}, please?')

        # creating a sequential model we will using it as a classifier
        self.classifier = self._creating_classifier(
            n_inputs=n_inputs, n_outputs=n_outputs, hidden_units=hidden_units, dropout=dropout)

        # Replace the model's classifier with the new classifier sequential layer
        if arch in ['resnet']: # , 'inception', 'googlenet'
            self.model.fc = self.classifier
        else:
            self.model.classifier = self.classifier

        # if GPU is available, move our model to it
        self.model = self.model.to(self.device)

        # Create a criterion object
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))

        # Create an optimizer, we will use SGD because it's better on this model than Adam
        if arch in ['resnet']: # , 'inception', 'googlenet'
            self.optimizer = optim.SGD(
                self.model.fc.parameters(), lr=lr)
        else:
            self.optimizer = optim.SGD(
                self.model.classifier.parameters(), lr=lr)

    # Training function
    def train(self, n_epochs, loaders, image_datasets, early_stopping=None):
        '''
        to train our CNN model architecture vgg on flowers training dataset
        returns a trained model
        '''
    
        # start time to calculate the time for training process
        print()
        print('#'+('='*50)+'#')
        print('Training.......')
        train_start = time()

        # Setting early stopping count
        early_stopping_count = 0

        # Setting model's best weights
        self.best_weights = self.model.state_dict()

        end_epoch = n_epochs + self.start_epoch - 1
        for epoch in range(self.start_epoch, n_epochs+1):
            with tqdm(total = len(image_datasets["training"])) as train_epoch:
                train_epoch.set_description(f'Train-> Epoch({epoch}/{n_epochs})')
                # epoch_start = time()

                # initialize variables to monitor training and validation loss
                # creating varibles with float 0 values to add on it training and validation loss
                train_loss = 0.0
                train_correct = 0.0
                train_total = 0.0

                valid_loss = 0.0
                valid_correct = 0.0
                valid_total = 0.0
                #=======================#
                #====train our model====#
                #=======================#
                self.model.train()
                for idx, (data, target) in enumerate(loaders['training']):
                    # to move it to GPU
                    X = data.to(self.device)
                    y = target.to(self.device)

                    # finding the loss and update the model parameters correctly and accordingly
                    # Set and clear the gradients of all optimized variables to zero
                    self.optimizer.zero_grad()
                    # forward pass
                    output = self.model(X)
                    # calculate the batch loss by using CrossEntropyLoss
                    loss = self.criterion(output, y)
                    # backward pass
                    loss.backward()
                    # to perform a single optimization step to update model parameters
                    self.optimizer.step()
                    
                    # update the training loss by this sample equation
                    train_loss = train_loss + ((1 / (idx + 1)) * (loss.data - train_loss))
                    
                    # to convert output probabilities to predicted class
                    _, pred = torch.max(output.data, 1)
                    
                    # comparing predictions with the true label
                    train_correct += np.sum(np.squeeze(pred.eq(y.data.view_as(pred))).cpu().numpy())
                    train_total += X.size(0)
                    
                    # Update the tqdm progress bar with Train loss and accuracy
                    desc = f'Train-> Epoch({epoch}/{end_epoch}) - Train loss = {train_loss:.4f} - Train Accuracy = {train_correct/train_total:.2%}'
                    train_epoch.set_description(desc)
                    train_epoch.update(X.shape[0])
                #========================#    
                #===validate our model===#
                #========================#
                self.model.eval()
            with tqdm(total = len(image_datasets["validation"])) as valid_epoch:
                valid_epoch.set_description(f'Valid-> Epoch{epoch}/{n_epochs}')
                for idx, (data, target) in enumerate(loaders['validation']):
                    X = data.to(self.device)
                    y = target.to(self.device)
            
                    # Updating the rate of validation loss
                    # forward pass
                    output = self.model(X)
                    # calculate the batch loss
                    loss = self.criterion(output, y)
                    
                    # update the average of validation loss by using this equation
                    valid_loss = valid_loss + ((1 / (idx + 1)) * (loss.data - valid_loss))
                    
                    # to convert output probabilities to predicted class
                    _, pred = torch.max(output.data, 1)
                    
                    # comparing predictions with the true label
                    valid_correct += np.sum(np.squeeze(pred.eq(y.data.view_as(pred))).cpu().numpy())
                    valid_total += X.size(0)
                    
                    # Updating the progress bar with valid loss and accuracy
                    desc = f'Valid-> Epoch({epoch}/{end_epoch} - Valid loss = {valid_loss:.4f} - Valid Accuracy = {valid_correct/(valid_total+1e-10):.2%}'
                    valid_epoch.set_description(desc)
                    valid_epoch.update(X.shape[0])


            # to append the train and valid loss for each epoch to the train_loss_list and valid_loss_list
            self.train_loss_list.append(train_loss.cpu().numpy())
            self.valid_loss_list.append(valid_loss.cpu().numpy())
            self.train_acc_list.append(100. * train_correct / train_total)
            self.valid_acc_list.append(100. * valid_correct / valid_total)

            # if validation loss has decreased, it'll save the model
            if valid_loss <= self.valid_loss_min:
                print(f'Validation loss decreased ({self.valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving the model ...')
                early_stopping_count = 0

                self.best_weights = self.model.state_dict()
                self.valid_loss_min = valid_loss
            else:
                early_stopping_count += 1

            if not early_stopping is None and early_stopping_count >= early_stopping:
                break

        self.model.load_state_dict(self.best_weights)
        self.class_to_idx = image_datasets['training'].class_to_idx

        self.start_epoch = epoch + 1
        # Save Model Summary to a pandas DataFrame and convert it to csv file
        print('Done: saving the model summary in DataFrame...')
        summary_dic = {
                'epoch': np.arange(1, self.start_epoch, 1),
                'train_losses': self.train_loss_list,
                'train_acc': self.train_acc_list,
                'valid_losses': self.valid_loss_list,
                'valid_acc': self.valid_acc_list,
        }
        self.model_summary = pd.DataFrame(summary_dic)

        train_end = time()
        print(f'Training total elapsed time: {train_end-train_start:.0f} Seconds')
        print('#'+('='*50)+'#')
        print()

    def save(self, save_dir='.'):
        """
        Saves the model as a checkpoint and it's attributes
        param:
        save_dir: the path to save the model at it
        """
        # Checkpoint file path
        now = datetime.now().strftime('%H%M%S') # to format the datename object in a specific format that is different from the standard format
        checkpoint_file = save_dir + '/' + self.arch + '_checkpoint_' + now + '.pth'
        # model state dictionary
        model_state = {
            'arch': self.arch,
            'state_dict': self.model.state_dict(),
            'classifier': self.classifier,
            'class_to_idx': self.class_to_idx,
            'optimizer': self.optimizer,
            'optimizer_dict': self.optimizer.state_dict(),
            'criterion': self.criterion,
            'criterion_dict': self.criterion.state_dict(),
            'model_summary': self.model_summary,
            'start_epoch': self.start_epoch,
            'valid_loss_min': self.valid_loss_min
        }
        # Save the model
        torch.save(model_state, checkpoint_file)
        print(f'Save model at {checkpoint_file}\n')

    def load(self, checkpoint_file):
        """
        Loads the model and it rebuilds the model
        param:
        checkpoint_file: a file path to load the model from it
        """
        # Load the model state from checkpoint file
        model_state = torch.load(checkpoint_file) # 
        self.arch = model_state['arch']
        self.model = models[self.arch]

        # Freezing the model parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Replace the model's classifier with the new classifier sequential layer
        self.classifier = model_state['classifier']
        if self.arch == 'resnet':
            self.model.fc = self.classifier
        else:
            self.model.classifier = self.classifier

        # if CUDA is available, move our model to GPU 
        self.model.to(self.device)

        # Load the model's state dict
        self.model.load_state_dict(model_state['state_dict'])

        # Load Class to index
        self.class_to_idx = model_state['class_to_idx']

        # Load optimizer and it's state dict
        self.optimizer = model_state['optimizer']
        self.optimizer.load_state_dict(model_state['optimizer_dict'])

        # Load criterion and it's state dict
        self.criterion = model_state['criterion']
        self.criterion.load_state_dict(model_state['criterion_dict'])

        # Load model's model_summary
        self.model_summary = model_state['model_summary']

        # Load start epoch
        self.start_epoch = model_state['start_epoch']

        # Load validation loss minimum
        self.valid_loss_min = model_state['valid_loss_min']

        # Load train and validation losses and accuracy
        self.train_losses = list(self.model_summary.train_losses)
        self.valid_losses = list(self.model_summary.valid_losses)
        self.train_acc = list(self.model_summary.train_acc)
        self.valid_acc = list(self.model_summary.valid_acc)

    def save_training_model_summary(self, summary_path):
        """
        it saves the training model_summary of the model
        and convert it from the Data Frame to csv file
        param:
        summary_path: a path to save the file at it
        """
        # convert from the DF to csv file
        self.model_summary.to_csv(summary_path, index=False)
        print(f'Saved training model_summary at {summary_path}\n')

    # Plotting the training model summary 
    def plot_model_summary(self):
        fig, ax = plt.subplots(figsize=(20, 6), ncols=2)
        ax[0].plot(self.model_summary.train_losses, color='r')
        ax[0].plot(self.model_summary.valid_losses, color='g')
        ax[0].spines['right'].set_visible(False) # to remove the axis that we don't need
        ax[0].spines['top'].set_visible(False)
        ax[0].set_ylabel('Loss', fontdict={'fontsize': 16})
        ax[0].set_xlabel('Epoch', fontdict={'fontsize': 16})
        ax[0].set_title('Training VS Validation Loss', fontdict={'fontsize': 24, 'fontweight':'bold'})
        ax[0].set_ylim(0)
        ax[0].legend(['Training', 'Validation'])
        ax[1].plot(self.model_summary.train_acc, color='b')
        ax[1].plot(self.model_summary.valid_acc, color='m')
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].set_ylabel('Accuracy', fontdict={'fontsize': 16})
        ax[1].set_xlabel('Epoch', fontdict={'fontsize': 16})
        ax[1].set_title('Training VS Validation Accuracy', fontdict={'fontsize': 24, 'fontweight':'bold'})
        ax[1].set_ylim(0)
        ax[1].legend(['Training', 'Validation'])
        plt.show()


    def perform_testing(self, dataloaders):
        ''' Evaluating and testing our model after training'''
        self.model.eval()
        
        total = 0
        accuracy = 0
        test_loss = 0
        total_images = len(dataloaders["testing"].batch_sampler) * dataloaders["testing"].batch_size
        for ii, (images, labels) in enumerate(dataloaders['testing']):
            X = images.to(self.device)
            y = labels.to(self.device)

            output = self.model.forward(X)
            # Calculating the loss
            # loss_fn = nn.NLLLoss()
            # test_loss += loss_fn(output, y).data[0]

            loss = self.criterion(output, y)
            test_loss = test_loss + ((1 / (ii + 1)) * (loss.data - test_loss))

            # Calculating the accuracy
            _, predicted = torch.max(output.data, 1)
            accuracy += (predicted == y).sum().item() 
            total += X.size(0)

        self.test_loss = test_loss / len(dataloaders['testing'])
        self.test_accuracy = accuracy / total
        
        print('Batch: {} '.format(ii+1),
              '\nTest Loss: {:.6f}.. '.format(test_loss),
              f'\nAccurately classified with: %.2f%% of {total_images} images. (%2d/%2d)\n' %
              (100. * self.test_accuracy, accuracy, total))
        
        if self.test_accuracy < 80/100:
            print('You can improve the accuracy by trying other arguments')
        else:
            print('Can you improve the accuracy?')

    def predict(self, image_file, topk=5, plot_predictions=False):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        
        # TODO: Implement the code to predict the class from an image file
        
        self.model.cpu()
        
        # evaluation mode
        # check: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval
        self.model.eval()
        
        # loading image as a torch Tensor
        image = process_image(image_file)
        
        # Unsqueeze returns a new tensor with a dimension of size one
        # check: https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        image = image.unsqueeze(0)
        
        # Stopping gradient calculation
        with torch.no_grad():
            output = F.softmax(self.model.forward(image), dim=1)
            top_probability, top_labels = torch.topk(output, topk)

            probability = np.squeeze(top_probability.numpy())

        idx_to_class = {self.class_to_idx[k]: k for k in self.class_to_idx}
        classes = [idx_to_class[i] for i in top_labels.numpy()[0]]

        if plot_predictions:
            self.plot_predicted_classes(image_file, probability, classes)

        return probability, classes

    def plot_predicted_classes(self, image_path, probabilty, classes):
        '''
        it plots the predictions and the top k(5) predicted classes.
        param:
        images: test images to plot
        labels: true labels
        model: the model to use for predictions
        topk: the number of top predicted classes to return
        '''
        # Displaying the images and the barchart with the predictions probabilty
        fig, axes = plt.subplots(1, 2, figsize=(16, 5)) # (16, 20)
        fig.subplots_adjust(hspace=0.2, wspace=0)

        predicted_label = classes[np.argmax(probabilty)]
        axes[0].imshow(Image.open(image_path))
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[0].set_title(f"prediction: {predicted_label}", fontdict={
                          'fontsize': 18, 'fontweight': 'bold'})

        axes[1].bar(classes, probabilty, color='c')
        axes[1].set_title(f"Top {len(classes)} predicted classes", fontdict={
                          'fontsize': 18, 'fontweight': 'bold'})
        axes[1].spines['top'].set_visible(False) # to delete the useless axis
        axes[1].spines['right'].set_visible(False) # to delete the useless axis

        prob_str = [f'{p:.2%}' for p in probabilty]
        for i, p in enumerate(probabilty):
            axes[1].text(i, p+.02, prob_str[i], ha='center')

        plt.show()

    def load_cat(self, path):
        """
        Loads category names file
        param: path: file path to load
        """
        with open(path, 'r') as f:
            self.cat_to_name = json.load(f)
