import pandas as pd
from transformers import InputExample, T5Tokenizer
from torch.utils.data import TensorDataset
import torch
import TweetNormalizer as normalizer

class TestDatasetLoader():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_data(self):
        sub_test_df = pd.read_csv('data/test_data.txt', delimiter='\t', header=None).rename(columns={0: 'tweet'})
        sub_test_df['label'] = -2
        return sub_test_df

    def load_dataset(self):
        df = self.load_data()

        examples = []
        labels = []
        from tqdm import tqdm
        for i, tweet in tqdm(df.iterrows()):
            
            text = normalizer.normalizeTweet(tweet['tweet'])
            
            label = tweet['label']
            guid = str(i)
            ex = InputExample(guid=guid, text_a=text, text_b=None, label=label)
            examples.append(ex)
            labels.append(label)
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [ex.text_a for ex in examples],
            padding="longest",
            max_length=150,
            pad_to_max_length = True,
            truncation=True,
            return_tensors="pt",
        )
        l = torch.tensor([ex.label for ex in examples])
        
        
        dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], l)
        return dataset, labels
    
