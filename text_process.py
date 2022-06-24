import torch
import os
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel


class TextProcessor:
    def __init__(self,
                 max_length: int = 50):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_length

    def __call__(self, text):
        encoded = self.tokenizer.batch_encode_plus([text], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        return description