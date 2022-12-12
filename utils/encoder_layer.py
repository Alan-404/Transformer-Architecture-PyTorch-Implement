import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from utils.multi_head_attention import MultiHeadAttention
from utils.position_wise_feed_forward_networks import PositionWiseFeedForwardNetworks
from utils.residual_connection import ResidualConnection
from typing import Union, Callable

class EncoderLayer(nn.Module):
    def __init__(self,d_model:int = 512, h: int = 8, d_ff: int = 2048, dropout_rate: float=0.1, eps: float=0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model = d_model, h = h)
        self.residual_connection_1 = ResidualConnection(dropout_rate=dropout_rate, eps=eps)

        self.ffn = PositionWiseFeedForwardNetworks(d_model= d_model, d_ff = d_ff, activation=activation)
        self.residual_connection_2 = ResidualConnection(dropout_rate=dropout_rate, eps= eps)

    def forward(self, x: Tensor, training: bool = False, mask: Tensor = None):
        multi_head_attention_output = self.multi_head_attention(x, x, x, mask)
        sublayer_1 = self.residual_connection_1(x, multi_head_attention_output, training)

        ffn_out = self.ffn(sublayer_1)
        sublayer_2 = self.residual_connection_2(sublayer_1, ffn_out, training)

        return sublayer_2