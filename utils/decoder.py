from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union
from utils.positional_encoding import PostionalEncoding
from utils.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, n: int = 6, h: int = 8, d_model: int = 512, d_ff: int = 2048, dropout_rate: float = 0.1, eps: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(Decoder, self).__init__()
        self.d_model = d_model

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_layer = PostionalEncoding()

        self.decoder_layers = [DecoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)]

        self.linear_output = nn.Linear(in_features=d_model, out_features=vocab_size)


    def forward(self, encoder_output: Tensor,  x:Tensor, training: bool = False, padding_mask: Tensor = None, look_ahead_mask: Tensor = None):
        length = x.size(1)
        embedding_output = self.embedding_layer(x)

        decoder_output = embedding_output + self.positional_layer.encode_position(length, self.d_model)

        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(encoder_output, x, training, padding_mask, look_ahead_mask)
        
        decoder_output = self.linear_output(decoder_output)

        return F.softmax(decoder_output)
