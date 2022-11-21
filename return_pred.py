import torch
import sys
from model import BertMNLIFinetuner
from dataset import AmazonDataset
import pandas as pd

def return_pred(review_to_predict):
    path_to_model = 'model.pt'
    bert = torch.load(path_to_model)
    bert.eval()
    
    review = {1: 'Positive review', 0: 'Negative review'}
    review_to_predict = pd.DataFrame(data = {'review_title': [review_to_predict], 'class_index': [None]})
    review_to_predict = AmazonDataset(review_to_predict)[0]
    with torch.no_grad():
        y_hat, _ = bert(review_to_predict[0], review_to_predict[1], review_to_predict[2])
        _, y_hat = torch.max(y_hat, dim=1)
    return review[int(y_hat)]

print(return_pred(str(sys.stdin)))
