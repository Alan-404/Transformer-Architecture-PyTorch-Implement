import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        
        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        dk = torch.tensor(k.size(0)).type(torch.float32)

        attention_scores = torch.matmul(q, k.transpose(2,3))/torch.sqrt(dk)

        if mask is not None:
            attention_scores += mask*(-1e10)

        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def splitting_head(self, x: Tensor):
        batch_size = x.size(0)
        length = x.size(1)
        d_model = x.size(2)

        heading_value = d_model // self.h

        tensor = torch.reshape(x, (batch_size, length, self.h, heading_value))
        tensor = torch.transpose(tensor, 1,2)

        return tensor

    def forward(self, v: Tensor, k: Tensor, q:Tensor, mask: Tensor = None):
        batch_size = q.size(0)
        length = q.size(1)

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        heading_q = self.splitting_head(qw)
        heading_k = self.splitting_head(kw)
        heading_v = self.splitting_head(vw)

        output, attention_weights = self.scaled_dot_product_attention(heading_q, heading_k, heading_v, mask)

        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (batch_size, length, self.d_model))

        output = self.linear_output(output)

        return output, attention_weights


    
        