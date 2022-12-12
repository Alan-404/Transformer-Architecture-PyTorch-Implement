from torch import Tensor
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate: float = 0.1, eps: float = 0.1):
        super(ResidualConnection, self).__init__()
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(eps)

    def forward(self, intial_input: Tensor, previous_output: Tensor, training: bool = False):
        if training:
            previous_output = self.dropout_layer(previous_output)

        addition_output = intial_input + previous_output

        return self.layer_norm(addition_output)   