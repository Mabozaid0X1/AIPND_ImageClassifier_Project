# Imports python modules
import torch
from classifier import Classifier
from utility_args import train_args, get_dataloaders
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    '''
    Main Function that used when file called from user terminal
    '''
    # Getting the inputs for user
    in_args = train_args()
    # Printing the arguments
    print('-'*52)
    print("Model training argument...")
    print('#'+('='*50)+'#')
    for arg in vars(in_args):
        print(f'{arg: <20}: {getattr(in_args, arg)}')
    
    print('#'+('='*50)+'#')
    print('-'*52)

    # if cuda is available, use GPU
    CUDA = 'cuda' if in_args.gpu and torch.cuda.is_available() else 'cpu'
    print('Predicted by ' + 'GPU' if CUDA == 'cuda' else 'CPU')
    print('-'*52)

    # Initializing the classifier class
    model_classifier = Classifier(CUDA)

    # Loading the data
    dataloaders, image_datasets, n_classes, weights = get_dataloaders(
        data_dir=in_args.data_dir, max_image_size=in_args.max_image_size, batch_size=in_args.batch_size)

    # Compiling the model if there is no checkpoint that available or Loading the model from checkpoint file.pth
    if in_args.check_point is None:
        model_classifier.compile(n_classes, arch=in_args.arch, hidden_units=in_args.hidden_units,
                dropout=in_args.dropout, lr=in_args.lr, weights=weights)
    else:
        model_classifier.load(in_args.check_point)

    # if not on evaluate only mode, Train and save the model 
    if not in_args.evaluate_only:
        model_classifier.train(in_args.epochs, dataloaders, image_datasets, early_stopping=in_args.early_stopping)
        model_classifier.save(in_args.save_dir)

    # In test data, perform testing and evaluating the model 
    model_classifier.perform_testing(dataloaders)

    # Plotting the model training summary if True
    if in_args.plot_summary:
        model_classifier.plot_model_summary()

    # Saving model training summary and converting from the pandas DataFrame to csv file if True
    if not in_args.save_summary is None:
        model_classifier.save_training_model_summary(in_args.save_summary)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)