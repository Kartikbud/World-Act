from torch import nn
import torch
import math

"""
This is where the individual components of the networks are defined,
more specifically the Attention Heads and MLPs that come together to
make transformers
"""

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass


# shape of the input embeddings: (B, N, d_model)
class MultiHeadAttention(nn.Module): # implemented with causal masking
    def __init__(self, 
                d_model: int = 192, 
                nheads : int = 16):

        super().__init__()
        self.d_model = d_model
        self.d_k = self.d_model // nheads
        self.H = nheads

        # Q V K projection weights
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
        
        #output layer
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)

    def attention(self, Q, K, V):
        # shape of Q K V are (B, H, N, d_k)
        N = Q.shape[2]
        # dot product attention operation
        scores = Q @ K.transpose(-2, -1) # shape: (B, H, N, N)
        scaled_scores =  scores / math.sqrt(self.d_k)

        # temporal masking by adding a mask before softmaxing
        mask = torch.tril(torch.ones(N, N, device=Q.device)).bool() 
        masked_scores = scaled_scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(masked_scores, dim=-1) # softmaxing to get weights
        outputs = weights @ V

        return outputs

    def forward(self, X): 
        # shape of X: (B, N, d_model)
        B = X.shape[0]
        N = X.shape[1]

        queries = self.W_q(X).reshape(B, N, self.H, self.d_k).permute(0, 2, 1, 3)
        keys = self.W_k(X).reshape(B, N, self.H, self.d_k).permute(0, 2, 1, 3)
        values = self.W_v(X).reshape(B, N, self.H, self.d_k).permute(0, 2, 1, 3)

        attn_output = self.attention(queries, keys, values)
        attn_output =  attn_output.permute(0, 2, 1, 3).reshape(B, N, self.d_model) # concatenating the output across the different heads

        output = self.W_o(attn_output)

        return output

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(x):
        pass