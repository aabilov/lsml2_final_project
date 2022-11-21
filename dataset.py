from transformers import BertModel, BertTokenizer
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class AmazonDataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df.class_index.to_list()
        self.texts = [tokenizer(text, 
                                padding='max_length', max_length = 64, truncation=True,
                                return_tensors="pt") for text in df['review_title']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        input_ids = batch_texts['input_ids']
        token_type_ids = batch_texts['token_type_ids']
        attention_mask = batch_texts['attention_mask']
        batch_y = self.get_batch_labels(idx)

        return input_ids, token_type_ids, attention_mask, batch_y
