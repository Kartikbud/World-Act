from torch import nn
import torch
import math

"""
This is where the individual components of the networks are defined,
more specifically the Attention Heads and MLPs that come together to
make transformers
"""

class TransformerBlock(nn.Module): # pre norm transformer implementation with adaptive layer norm as an option
    def __init__(self, 
                d_model : int = 192,
                nheads : int = 16,
                d_action : int = None,
                dropout : float = 0.1,
                causal_masking : bool = True,
                adaptive : bool = True):
        super().__init__()

        self.d_model = d_model
        self.nheads = nheads
        self.d_action = d_action
        self.hidden_dim = 4 * self.d_model
        self.dropout = dropout
        self.adaptive = adaptive
        
        if self.adaptive:
            self.W_LNparams = nn.Linear(in_features=d_action, out_features=4*d_model) # 2 sets of parameters for each norm
            # initing the weights and bias to 0
            self.W_LNparams.weight.data.zero_()
            self.W_LNparams.bias.data.zero_()
            self.norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        else:
            self.norm = nn.LayerNorm(self.d_model, elementwise_affine=True)

        self.attn_block = MultiHeadAttention(d_model=d_model,
                                             nheads=nheads,
                                             dropout=self.dropout,
                                             causal_masking=causal_masking)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.hidden_dim, out_features=self.d_model),
            nn.Dropout(self.dropout)
        )                                        

    def forward(self, z, a=None): # positional embeddings should come into this
        # Note: can also add gates to make training more stable if need be
        if self.adaptive:
            params = self.W_LNparams(a)
            scale_one, shift_one, scale_two, shift_two = params.chunk(4, dim=-1) # splitting the tensor into 2 seperate tensors for each param
            residual_one = self.attn_block((1 + scale_one) * self.norm(z) + shift_one)
        else:
            residual_one = self.attn_block(self.norm(z))
        z = z + residual_one
        
        if self.adaptive:
            residual_two = self.mlp((1 + scale_two) * self.norm(z) + shift_two)
        else:
            residual_two = self.mlp(self.norm(z))
        z = z + residual_two

        return z


# shape of the input embeddings: (B, N, d_model)
class MultiHeadAttention(nn.Module): # implemented with causal masking
    def __init__(self, 
                d_model : int = 192, 
                nheads : int = 16,
                dropout : float = 0.1,
                causal_masking : bool = True):

        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.d_k = self.d_model // nheads
        self.H = nheads
        self.causal_masking = causal_masking

        self.drop = nn.Dropout(p=self.dropout)

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
        if self.causal_masking:
            mask = torch.tril(torch.ones(N, N, device=Q.device)).bool() 
            masked_scores = scaled_scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(masked_scores, dim=-1) # softmaxing to get weights
        else:
            weights = torch.softmax(scaled_scores, dim=-1) # softmaxing to get weights
        weights = self.drop(weights)
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
        output = self.drop(output)

        return output