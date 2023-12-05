import pandas as pd
from transformers import InputExample, T5Tokenizer
from torch.utils.data import TensorDataset
import torch
import TweetNormalizer as normalizer

class DatasetLoader():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_data(self):
        pos = pd.read_csv('data/train_pos.txt', delimiter='\t', header=None).sample(10000, random_state=42)
        pos['label'] = 1
        neg = pd.read_csv('data/train_neg.txt', delimiter='\t', header=None).sample(10000, random_state=42)
        neg['label'] = 0

        train_df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        dev_df = train_df.sample(frac=0.2, random_state=42).rename(columns={0: 'tweet'})
        train_df = train_df.drop(dev_df.index).rename(columns={0: 'tweet'})
        test_df = pd.read_csv('data/test_data.txt', delimiter='\t', header=None).rename(columns={0: 'tweet'})
        test_df['label'] = -2
        test_df = test_df
        return train_df, dev_df, test_df

    def load_dataset(self, split="train"):
        train, dev, test = self.load_data()
        if split == 'train':
            df = train
        elif split == 'dev':
            df = dev
        elif split == 'test': 
            df = test
        else: 
            raise ValueError("split should be in [train, dev, test]")    
        
        examples = []
        labels = []
        for i, tweet in df.iterrows():
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
        tokenized_targets = {'input_ids': l,
                             'attention_mask' : torch.ones_like(l)}
        
        dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'],
                                tokenized_targets['input_ids'], tokenized_targets['attention_mask'] 
                            )
        return dataset, labels
    
