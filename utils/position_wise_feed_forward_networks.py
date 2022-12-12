import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable
import torch
class PositionWiseFeedForwardNetworks(nn.Module):
    def __init__(self,d_model: int =512, d_ff:int=2048, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu):
        super(PositionWiseFeedForwardNetworks, self).__init__()
        self.hidden_layer = nn.Linear(in_features=d_model, out_features=d_ff)
        self.activation = activation
        self.output_layer = nn.Linear(in_features=d_ff, out_features=d_model)
    def forward(self, x: torch.Tensor):
        x = self.hidden_layer(x)
        x = self.activation(x)
        return self.output_layer(x)