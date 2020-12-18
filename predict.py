# Imports python modules
import torch
from classifier import Classifier
from utility_args import predict_args
import warnings
import json
warnings.filterwarnings('ignore')

def main():
    '''
    Main Function that used when file called from user terminal
    '''
    # Getting the inputs for user
    in_args = predict_args()
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


    # Loading the model from checkpoint file.pth
    model_classifier.load(in_args.check_point)

    # Loading category names if it's available
    if not in_args.category_names is None:
        model_classifier.load_cat(in_args.category_names)

    with open(in_args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Predicting the image
    top_prob, top_classes = model_classifier.predict(
        in_args.input, topk=in_args.top_k, plot_predictions=in_args.plot_predictions)

    label = top_classes[0]
    prob = top_prob[0]

    print(f'\nPrediction\n---------------------------------')

    print(f'Flower      : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}%')

    print(f'\nTop K\n---------------------------------')

    for i in range(len(top_prob)):
        print(
            f"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")

if __name__ == '__main__':
    main()
