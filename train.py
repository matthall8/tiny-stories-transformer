#%%
import torch
import torch.backends.cudnn as cudnn
from dataset import StoryDataset
from model import MatTransformer
from torch.nn.utils.rnn import pad_sequence
import wandb
import random

#%%
torch.manual_seed(42)
#%%
def collate_fn(batch):
    source, target = zip(*batch)
    source_padded = pad_sequence(source, batch_first=True, padding_value=0) 
    target_padded = pad_sequence(target, batch_first=True, padding_value=0)
    return source_padded, target_padded

full_dataset = StoryDataset()
#%%
max_len = full_dataset.getMaxLength()
vocab_size = full_dataset.getVocabSize()
version = 4

# hyperparameters 
batch_size = 32
n_embd = 128
epochs = 5
ff_size = 48
learning_rate = 0.001
dropout_rate = 0.3
n_layer = 4
n_heads = 4

full_training_set = torch.utils.data.DataLoader(full_dataset, batch_size = batch_size, collate_fn=collate_fn, shuffle=True)

#%%
m = MatTransformer(max_len, n_embd, vocab_size, ff_size, dropout_rate, n_heads, n_layer)
opt = torch.optim.Adam(m.parameters(), lr=learning_rate)
total_params = sum(p.numel() for p in m.parameters())

# wandb tracking

wandb.init(
    # set the wandb project where this run will be logged
    project="tiny-story-v2",
    
    # track hyperparameters and run metadata
    config= {
    "learning_rate": learning_rate,
    "architecture": "MatTransformer - 4 masked blocks & Adam",
    "parameters": total_params,
    "vocab_size": vocab_size,
    "ff_size": ff_size,
    "batch_size": batch_size,
    "dropout_rate": dropout_rate,
    "dataset": "Tiny Story - 888",
    "epochs": epochs
    }
)

print("Parameters:", total_params)
for epoch in range(epochs):
  for idx, (source, target) in enumerate(full_training_set):
    m.train()
    x = source
    y = target
    p = m(x)
    p_class = p.permute(0, 2, 1)
    l = torch.nn.functional.cross_entropy(p_class, y, ignore_index=0)
    wandb.log({"Loss": l.item()})
    if idx % 1000 == 0: 
       print("Loss:", l.item())
       torch.save(m, f'models/transformer_00{version}_{idx}.pth')
    l.backward()
    opt.step()
    opt.zero_grad()

wandb.finish()
torch.save(m, f'models/transformer_00{version}_final.pth')
# %%