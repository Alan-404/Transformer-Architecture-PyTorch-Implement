import torch
import os
import io
from data.tokenizer import Tokenizer
from torch.utils.data import DataLoader, TensorDataset

class DatasetBuilder:
    def __init__(self, inp_lang: str, targ_lang: str, tokenizers_folder: str):
        self.inp_lang = inp_lang
        self.targ_lang = targ_lang

        self.tokenizers_folder = tokenizers_folder

        self.inp_tokenizer = Tokenizer()
        self.targ_tokenizer = Tokenizer()

    def build_tokenizer(self, tokenizer: Tokenizer, data: list):
        tokenizer.fit_to_texts(data)
        return tokenizer

    def tokenize(self, tokenizer: Tokenizer, data: list, max_length: int):
        sequences = tokenizer.texts_to_sequences(data)
        return tokenizer.pad_sequences(sequences, maxlen=max_length)
    
    def build_dataset(self, inp_data_path: str, targ_data_path: str, batch_size: int, num_data: int, max_length: int = 40):
        inp_data = io.open(inp_data_path, encoding='utf-8').read().strip().split('\n')
        targ_data = io.open(targ_data_path, encoding='utf-8').read().strip().split('\n')

        if num_data is not None:
            inp_data = inp_data[:num_data]
            targ_data = targ_data[:num_data]

        self.inp_tokenizer =  self.build_tokenizer(self.inp_tokenizer, inp_data)
        self.targ_tokenizer = self.build_tokenizer(self.targ_tokenizer, targ_data)

        inp_sequences = self.tokenize(self.inp_tokenizer, inp_data, max_length)
        targ_sequences = self.tokenize(self.targ_tokenizer, targ_data, max_length)

        dataset = TensorDataset(torch.Tensor(inp_sequences).type(torch.int64), torch.Tensor(targ_sequences).type(torch.int64))
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataset_loader