
from Trainer import Classifier
import os
import argparse

from load_data import DatasetLoader
import pandas as pd

DATA_FOLDER = 'data/eval' 

def evaluate(model_dir, per_gpu_eval_batch_size):
    language_model = model_dir.split("/")[-1].replace("_", "/")
    classifier = Classifier(output_model_dir=model_dir, 
                            cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
                            pretrained_model_name_or_path=language_model)
    
    print(language_model)
    preds = classifier.predict(per_gpu_eval_batch_size=per_gpu_eval_batch_size)
    
    if isinstance(preds, list):
        labels = [pred for pred in preds]
    else: 
        labels = [pred.item() for pred in preds ]
    labels = [-1 if label == 0 else label for label in labels]  
    ids = [i for i in range(1, len(labels)+1)]
    print("Saving at: ", f'submit_{language_model.split("/")[-1]}.csv')
    pd.DataFrame(zip(ids, labels)).to_csv(f'submit_{language_model.split("/")[-1]}.csv', index=False, header=['Id', 'Prediction'])
    
def main():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model-dir', dest='model_dir', required=True,
                        help='the folder/google bucket in which the model will be stored or loaded from.')
    
    parser.add_argument('--per_gpu_eval_batch_size', default=16, type=int)
    args = parser.parse_args()
    
    evaluate(model_dir=args.model_dir, per_gpu_eval_batch_size=args.per_gpu_eval_batch_size)

if __name__ == '__main__':
    main()