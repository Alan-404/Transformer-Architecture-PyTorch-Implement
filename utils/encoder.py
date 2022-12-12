from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable
from utils.encoder_layer import EncoderLayer
from utils.positional_encoding import PostionalEncoding

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, n: int = 6, h: int = 8, d_model: int = 512, d_ff: int = 2048, dropout_rate: float = 0.1, eps: float= 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.positional_layer = PostionalEncoding()

        self.encoder_layers = [EncoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)]

    def forward(self, x: Tensor, training: bool = False, mask: Tensor = None):
        length = x.size(1)

        embedding_output = self.embedding_layer(x)

        encoder_output = embedding_output + self.positional_layer.encode_position(length=length, embeded_dim=self.d_model)

        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(x, training, mask)

        return encoder_output