import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Callable
from utils.encoder import Encoder
from utils.decoder import Decoder
from torch.utils.data import DataLoader
import torch.optim as optim

class Transformer(nn.Module):
    def __init__(self, inp_vocab_size: int, targ_vocab_size: int, n: int = 6, h:int =8, d_model: int = 512, d_ff: int = 2048, dropout_rate: float=0.1, eps: float=0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(Transformer, self).__init__()
        self.encoder = Encoder(inp_vocab_size, n, h, d_model, d_ff, dropout_rate, eps, activation)
        self.decoder = Decoder(targ_vocab_size, n, h, d_model, d_ff, dropout_rate, eps, activation)

    def forward(self, encoder_in: Tensor, decoder_in: Tensor, training: bool = False, encoder_padding_mask: Tensor = None, decoder_padding_mask: Tensor =None, look_ahead_mask: Tensor = None):
        encoder_output = self.encoder(encoder_in, training, encoder_padding_mask)

        decoder_output = self.decoder(encoder_output, decoder_in, training, decoder_padding_mask, look_ahead_mask)

        return decoder_output

