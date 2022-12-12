import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer import Transformer
from typing import Callable, Union
from torch import Tensor
from torch.utils.data import DataLoader

class TransformerModule:
    def __init__(self, inp_vocab_size: int, targ_vocab_size: int, n: int = 6, h: int = 8, d_model: int = 512, d_ff: int = 2048, dropout_rate: float= 0.1, eps: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        self.model = Transformer(inp_vocab_size, targ_vocab_size, n, h, d_model, d_ff, dropout_rate, eps, activation)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)

    def fit(self, dataloader: DataLoader, epochs: int = 1):
        for epoch in range(epochs):
            running_loss = 0.0

            for i, data in enumerate(dataloader, 0):
                inp, targ = data

                self.optimizer.zero_grad()

                outputs = self.model(inp)

                loss = self.criterion(outputs, targ)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0