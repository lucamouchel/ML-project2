import argparse
import os
import sys

sys.path.append(".")
from Trainer import Classifier

DATA_FOLDER = 'data'
def training(train_batch_size,
             learning_rate,
             epochs,
             language_model,
             grad_acc):

    classifier = Classifier(
            output_model_dir=f"models/{language_model.replace('/', '_')}",
            cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
            pretrained_model_name_or_path=language_model)
    
    classifier.train(train_batch_size=train_batch_size,
                        learning_rate=learning_rate,
                        num_train_epochs=epochs,
                        gradient_accumulation_steps=grad_acc)

def parse_args():
    """
    Parsing input arguments which makes the training process customizable. 
    We have default arguments as well.
    """
    parser = argparse.ArgumentParser(description='train a binary classifier')

    parser.add_argument('--language-model', default='cardiffnlp/twitter-roberta-base-sentiment-latest', help='Can be either some huggingface model or a '
                                                                         'path to a model. If the path is in GCS we '
                                                                         'download it first.')
    
    parser.add_argument('--epochs', default=5,help='number of epochs to train')
    parser.add_argument('--batch-size', default=8, help='batch size')
    parser.add_argument('--val-batch-size', default=8, help='validation batch size')
    parser.add_argument('--lr', default=8e-5, help='learning rate')

    parser.add_argument('--gradient-accumulation', default=2)
    return parser.parse_args()

def main():
    """
    launch the training process here
    """
    args = parse_args()
    training(train_batch_size=int(args.batch_size),
                epochs=int(args.epochs),
                learning_rate=float(args.lr),
                language_model=args.language_model,
                grad_acc=int(args.gradient_accumulation))
    
    
if __name__ == '__main__':
    main()