import sys
sys.path.append(".")
import os
import argparse

import pandas as pd
from sklearn import metrics

def load_predictions(model_name):
    MODEL_NAME = model_name.replace("/", "_")
    predicted_values = pd.read_csv("submit_test_" + MODEL_NAME + ".csv")
    true_values = pd.read_csv(os.path.join("validation_sets", MODEL_NAME, "validation_set.csv"), index_col=0, names=['Id','Prediction'])
    true_values[true_values['Prediction']==0] = -1
    return true_values, predicted_values

def calculate_metrics(true, pred):
    print(len(pred.values))
    accuracy = metrics.accuracy_score(true.values, pred.values)
    f1 = metrics.f1_score(true.values, pred.values)
    return f1, accuracy

def evaluate(model_name):
    true, pred = load_predictions(model_name)
    f1, accuracy = calculate_metrics(true.Prediction, pred.Prediction)
    print(f"Model: {model_name},\n \t F1 score : {round(f1,4)},\n \t accuracy : {round(accuracy,4)}")
    
def main():
    parser = argparse.ArgumentParser(description='Validate tested model')
    parser.add_argument('--model-name', dest='model_name', required=True,
                        help='the base name of the model that you trained')

    args = parser.parse_args()
    
    evaluate(model_name=args.model_name)

if __name__ == '__main__':
    main()