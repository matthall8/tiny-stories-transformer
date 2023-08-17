#%%
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np

#%%
torch.manual_seed(42)
#%%
class Head(nn.Module):
    def __init__(self, n_embds, n_heads, masked, dropout_rate):
        super(Head, self).__init__()
        self.n_embds = n_embds
        self.n_heads = n_heads
        self.masked = masked
        self.dropout_rate = dropout_rate
        head_size = n_embds // n_heads
        self.key_linear_layer = nn.Linear(n_embds, head_size, bias=False)
        self.query_linear_layer = nn.Linear(n_embds, head_size, bias=False)
        self.value_linear_layer = nn.Linear(n_embds, head_size, bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input):
        k = self.key_linear_layer(input)
        q = self.query_linear_layer(input)
        k_dims = k.shape[2]
        k_transpose = k.permute(0,2,1)
        scores = torch.bmm(q, k_transpose)
        scaled_scores = scores / (k_dims**0.5)
        if self.masked:
            masked_scores = self.apply_attention_mask(scaled_scores)
            softmax_scores = F.softmax(masked_scores, dim=2)
        else: 
            softmax_scores = F.softmax(scaled_scores, dim=2)
        softmax_dropout = self.dropout(softmax_scores)
        v = self.value_linear_layer(input)
        output = torch.bmm(softmax_dropout, v)
        return output
     
    def apply_attention_mask(self, attention_scores):
        # Generate a mask for the lower triangular part of each matrix in the batch
        batch_size = attention_scores.size(0)
        size = attention_scores.size(1)
        mask = torch.tril(torch.ones(batch_size, size, size), diagonal=0)
    
        # Create a tensor of -inf values with the same shape as attention_scores
        negative_inf = torch.full_like(attention_scores, float('-inf'))
    
        # Use torch.where to fill masked positions with -inf
        masked_attention_scores = torch.where(mask.bool(), attention_scores, negative_inf)
    
        return masked_attention_scores
     
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embds, n_heads, masked, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.n_embds = n_embds
        self.n_heads = n_heads
        self.masked = masked
        self.dropout_rate = dropout_rate
        self.heads = nn.ModuleList([Head(n_embds, n_heads, masked, dropout_rate) for _ in range (n_heads)])
        self.proj = nn.Linear(n_embds, n_embds)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, n_embds, n_heads,dropout_rate,ff_size):
        # n_embds: embedding dimension, n_heads: the number of heads we'd like
        super(TransformerBlock, self).__init__()
        self.n_embds = n_embds
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.ff_size = ff_size
        self.sa = MultiHeadAttention(n_embds, n_heads, masked=True, dropout_rate=dropout_rate)
        self.ffwd = FeedForward(n_embds,ff_size,dropout_rate)
        self.ln1 = nn.LayerNorm(n_embds)
        self.ln2 = nn.LayerNorm(n_embds)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MatTransformer(nn.Module):
    def __init__(self,max_sequence_len, n_embds, vocab_size, ff_size, dropout_rate, n_heads, n_layers):
        super(MatTransformer, self).__init__()
        self.n_embds = n_embds
        self.dropout_rate = dropout_rate
        self.ff_size = ff_size
        self.n_heads = n_heads
        self.embeddings = nn.Embedding(vocab_size,n_embds)
        self.positional_encodings = self.get_positional_encoding(max_sequence_len, n_embds) 
        self.blocks = nn.Sequential(*[TransformerBlock(n_embds, n_heads,dropout_rate,ff_size) for _ in range(n_layers)])
        self.layer_norm_final = nn.LayerNorm(n_embds)
        self.output_linear_layer = nn.Linear(n_embds, vocab_size)
    
    def forward(self, x):
        # embeddings and pos encodings
        embeddings = self.embeddings(x)
        pos_encodings = self.positional_encodings[:x.shape[1],:]
        emb = embeddings + pos_encodings
        transformer_block_output = self.blocks(emb)
        final_layer_norm =  self.layer_norm_final(transformer_block_output)
        output = self.output_linear_layer(final_layer_norm)

        return output

    def get_positional_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    pos_enc[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
        return pos_enc

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForward, self).__init__()
        self.d_model =d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(p=dropout_rate),     
        )

    def forward(self, x):
        return self.ff(x)
# %%