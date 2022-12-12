from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union
from utils.multi_head_attention import MultiHeadAttention
from utils.position_wise_feed_forward_networks import PositionWiseFeedForwardNetworks
from utils.residual_connection import ResidualConnection

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout_rate: float = 0.1, eps: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super(DecoderLayer, self).__init__()

        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.residual_connection_1 = ResidualConnection(dropout_rate=dropout_rate, eps= eps)

        self.multi_head_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.residual_connection_2 = ResidualConnection(dropout_rate=dropout_rate, eps=eps)

        self.ffn = PositionWiseFeedForwardNetworks(d_model=d_model, d_ff=d_ff, activation=activation)
        self.residual_connection_3 = ResidualConnection(dropout_rate=dropout_rate, eps=eps)

    def forward(self, encoder_output: Tensor,  x: Tensor, training: bool = False, padding_mask: Tensor = None, look_ahead_mask: Tensor = None):
        masked_multi_head_attention_output = self.masked_multi_head_attention(x, x, x, look_ahead_mask)
        sublayer_1 = self.residual_connection_1(x, masked_multi_head_attention_output, training)

        multi_head_attention_output = self.multi_head_attention(encoder_output, encoder_output, sublayer_1, padding_mask)
        sublayer_2 = self.residual_connection_2(sublayer_1, multi_head_attention_output, training)

        ffn_output = self.ffn(sublayer_2)
        sublayer_3 = self.residual_connection_3(sublayer_2, ffn_output, training)

        return sublayer_3