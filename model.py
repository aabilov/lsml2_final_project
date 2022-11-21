from transformers import BertModel, BertTokenizer
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class BertMNLIFinetuner(pl.LightningModule):

    def __init__(self):
        super(BertMNLIFinetuner, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
        
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2


    def forward(self, input_ids, attention_mask, token_type_ids):
        
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        token_type_ids = token_type_ids.squeeze(1)
          
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        attn = out['attentions']
        h = out['last_hidden_state']
        
        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn

    def training_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, label = batch
         
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        loss = F.cross_entropy(y_hat, label)
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        input_ids, attention_mask, token_type_ids, label = batch
         
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        loss = F.cross_entropy(y_hat, label)
        
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())
        self.log_dict({'test_acc': torch.tensor(test_acc)})
        return {'test_acc': torch.tensor(test_acc)}
        
    def test_epoch_end(self, outputs):

        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        self.log_dict(tensorboard_logs)
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=1e-05, eps=1e-08)


    def train_dataloader(self):
        return train_dataloader


    def val_dataloader(self):
        return val_dataloader


    def test_dataloader(self):
        return test_dataloader
