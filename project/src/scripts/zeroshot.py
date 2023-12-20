from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from load_data import DatasetLoader
import sys
sys.path.append(".")
from project.src.utils import metrics

data_loader = DatasetLoader(None)
train, dev, test = data_loader.load_data()  
val_set = data_loader.load_validation_dataset() 

from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

def classify_roberta():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True, max_length=512)

    labels = val_set['label'].tolist()
    preds = []
    predictions = pipe(val_set['tweet'].tolist())
    for pred in predictions:
        preds.append(1 if pred['label'] == 'positive' or pred['label'] == 'neutral' else 0)
    
    

    print(metrics.compute(preds, labels))

def classify_bart():
    import numpy as np
    from tqdm import tqdm
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['negative', 'positive']
    
    labels = val_set['label'].tolist()
    preds = []
    for tweet in tqdm(val_set['tweet'].tolist()):
        sequence_to_classify = tweet
        pred = classifier(sequence_to_classify, candidate_labels)
        preds.append(np.argmax(pred['scores']))

    print(metrics.compute(preds, labels))
classify_bart()