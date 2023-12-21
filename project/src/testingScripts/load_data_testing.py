import pandas as pd
from transformers import InputExample, T5Tokenizer
from torch.utils.data import TensorDataset
import torch
import sys
sys.path.append(".")
import project.src.utils.TweetNormalizer as normalizer

class DatasetLoader():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_data(self):
        """
        Loads in data from data/train_pos.txt and data/train_neg.txt
        """
        # Loading in positive and negative samples
        pos = pd.read_csv('data/train_pos.txt', header=None, delimiter='\t').sample(11000, random_state=4)
        pos['label'] = 1
        neg = pd.read_csv('data/train_neg.txt', header=None, delimiter='\t').sample(11000, random_state=4)
        neg['label'] = 0
        
        train_test_split = 0.8 # 80% of data to training, 20% to testing

        # Concatenation of positive and negative samples
        full_df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        
        train_df = full_df[:int(len(full_df)*train_test_split)]
        # Subsection of training to validate DURING training
        dev_df = train_df.sample(frac=0.2, random_state=42).rename(columns={0: 'tweet'})
        # Removing dev_df lines from train_df
        train_df = train_df.drop(dev_df.index).rename(columns={0: 'tweet'})

        # Creating test df from remaining lines
        test_df = full_df[int(len(full_df)*train_test_split):].rename(columns={0: 'tweet'})
        # Validation is copy of test_df with the labels
        validation_df = test_df.copy()
        test_df['label'] = -2
        test_df = test_df
        return train_df, dev_df, test_df, validation_df

    def load_dataset(self, split="train"):
        """
        Calls in dataset depending on split that was asked for
        """
        train, dev, test, validation = self.load_data()
        if split == 'train':
            df = train
        elif split == 'dev':
            df = dev
        elif split == 'test': 
            df = test
        elif split == 'validation':
            df = validation
        else: 
            raise ValueError("split should be in [train, dev, test]")    
        
        if split != 'validation':
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
        else:
          return df
    
