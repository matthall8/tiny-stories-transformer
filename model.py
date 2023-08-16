#%%
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np

#%%
torch.manual_seed(42)
#%%
class MultiHeadAtt(nn.Module):
     def __init__(self, n_embd):
        super(MultiHeadAtt, self).__init__()
        self.n_embd = n_embd
        self.head_input = n_embd // 2
        self.key_01_linear_layer = nn.Linear(self.head_input, self.head_input)
        self.query_01_linear_layer = nn.Linear(self.head_input, self.head_input)
        self.value_01_linear_layer = nn.Linear(self.head_input, self.head_input)
        self.key_02_linear_layer = nn.Linear(self.head_input, self.head_input)
        self.query_02_linear_layer = nn.Linear(self.head_input, self.head_input)
        self.value_02_linear_layer = nn.Linear(self.head_input, self.head_input)
        self.final_linear_layer = nn.Linear(n_embd, n_embd)

     def forward(self, input, masked):
        input_head_01 = input[:,:,:self.head_input]
        input_head_02 = input[:,:,self.head_input:]
        k_01 = self.key_01_linear_layer(input_head_01)
        v_01 = self.value_01_linear_layer(input_head_01)
        q_01 = self.query_01_linear_layer(input_head_01)
        k_02 = self.key_02_linear_layer(input_head_02)
        v_02 = self.value_02_linear_layer(input_head_02)
        q_02 = self.query_02_linear_layer(input_head_02)
        k_dims = k_01.shape[2]
        k_01_transpose = k_01.permute(0,2,1)
        k_02_transpose = k_02.permute(0,2,1)
        scores_01 = torch.bmm(q_01, k_01_transpose)
        scores_02 = torch.bmm(q_02, k_02_transpose)
        scaled_scores_01 = scores_01 / (k_dims**0.5)
        scaled_scores_02 = scores_02 / (k_dims**0.5)
        if masked:
            masked_scores_01 = self.apply_attention_mask(scaled_scores_01)
            masked_scores_02 = self.apply_attention_mask(scaled_scores_02)
            softmax_scores_01 = F.softmax(masked_scores_01, dim=2)
            softmax_scores_02 = F.softmax(masked_scores_02, dim=2)
        else: 
            softmax_scores_01 = F.softmax(scaled_scores_01, dim=2)
            softmax_scores_02 = F.softmax(scaled_scores_02, dim=2)
        head_output_01 = torch.bmm(softmax_scores_01, v_01)
        head_output_02 = torch.bmm(softmax_scores_02, v_02)
        concatenated = torch.cat([head_output_01, head_output_02], dim=2) 
        output = self.final_linear_layer(concatenated)
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

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(FeedForward, self).__init__()
        self.d_model =d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.d_ff, self.d_model),
            nn.ReLU(),  
            nn.Dropout(p=dropout_rate),          
        )

    def forward(self, x):
        return self.ff(x)

class MatTransformer(nn.Module):
    def __init__(self,max_sequence_len, n_embd, vocab_size, ff_size, dropout_rate):
        super(MatTransformer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size,n_embd)
        self.positional_encodings = self.get_positional_encoding(max_sequence_len, n_embd) 
        self.n_embd = n_embd
        self.ff_size = ff_size
        self.multi_head_attention_layer_01 = MultiHeadAtt(n_embd)
        self.multi_head_attention_layer_02 = MultiHeadAtt(n_embd)
        self.multi_head_attention_layer_03 = MultiHeadAtt(n_embd)
        self.multi_head_attention_layer_04 = MultiHeadAtt(n_embd)
        self.norm_layer_01 = nn.LayerNorm(n_embd)
        self.norm_layer_02 = nn.LayerNorm(n_embd)
        self.norm_layer_03 = nn.LayerNorm(n_embd)
        self.norm_layer_04 = nn.LayerNorm(n_embd)
        self.norm_layer_05 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, ff_size, dropout_rate)
        self.output_linear_layer = nn.Linear(n_embd, vocab_size)
    def forward(self, x):
        # embeddings and pos encodings
        embeddings = self.embeddings(x)
        pos_encodings = self.positional_encodings[:x.shape[1],:]
        emb = embeddings + pos_encodings
        # first transformer block with mask single head attention layer
        multi_head_attention_01 = self.multi_head_attention_layer_01(emb, masked=True)
        add_layer_01 = emb + multi_head_attention_01
        layer_norm_01 = self.norm_layer_01(add_layer_01)
        # second transformer block
        multi_head_attention_02 = self.multi_head_attention_layer_02(layer_norm_01, masked=True)
        add_layer_02 = layer_norm_01 + multi_head_attention_02
        layer_norm_02 = self.norm_layer_02(add_layer_02)
        # third transformer block
        multi_head_attention_03 = self.multi_head_attention_layer_03(layer_norm_02, masked=True)
        add_layer_03 = layer_norm_02 + multi_head_attention_03
        layer_norm_03 = self.norm_layer_03(add_layer_03)
        # fourth transformer block
        multi_head_attention_04 = self.multi_head_attention_layer_04(layer_norm_03, masked=True)
        add_layer_04 = layer_norm_03 + multi_head_attention_04
        layer_norm_04 = self.norm_layer_04(add_layer_04)
        # feed forward 
        feed_forward = self.feed_forward(layer_norm_04)
        add_layer_05 = layer_norm_04 + feed_forward
        layer_norm_05 = self.norm_layer_04(add_layer_05)
        # linear output layer
        output_linear_layer = self.output_linear_layer(layer_norm_05)
        return output_linear_layer

    def get_positional_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                else:
                    pos_enc[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
        return pos_enc  
# %%