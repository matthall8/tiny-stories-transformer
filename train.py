#%%
import torch
import torch.backends.cudnn as cudnn
from dataset import StoryDataset, Tokenizer
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
n_embd = 512
epochs = 5
ff_size = 48
learning_rate = 0.001
dropout_rate = 0.3
n_layer = 8
n_heads = 4

full_training_set = torch.utils.data.DataLoader(full_dataset, batch_size = batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True, )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
m = MatTransformer(max_len, n_embd, vocab_size, ff_size, dropout_rate, n_heads, n_layer, device=device)
m = m.to(device)
# opt = torch.optim.Adam(m.parameters(), lr=learning_rate)
opt = bnb.optim.Adam8bit(m.parameters(), lr=learning_rate)
total_params = sum(p.numel() for p in m.parameters())

# wandb tracking

""" wandb.init(
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
 """
print("Parameters:", total_params)
for epoch in range(epochs):
  for idx, (source, target) in enumerate(full_training_set):
    m.train()
    x = source.to(device)
    y = target.to(device)
    p = m(x)
    p_class = p.permute(0, 2, 1)
    l = torch.nn.functional.cross_entropy(p_class, y, ignore_index=0)
    # wandb.log({"Loss": l.item()})
    if idx % 1000 == 0: 
       print("Loss:", l.item())
       torch.save(m, f'models/transformer_00{version}_{idx}.pth')
    l.backward()
    l = l.cpu()
    opt.step()
    opt.zero_grad()

# wandb.finish()
torch.save(m, f'models/transformer_00{version}_final.pth')
# %%

def generate_text(model, tokenizer, seed_text, max_length=1000):
    model.eval()  # Set model to evaluation mode
    generated_text = seed_text
    encoded_input = tokenizer.encode(seed_text)

    for _ in range(max_length - len(seed_text)):
        input_tensor = torch.tensor([encoded_input], device=device) # Convert to tensor and move to device
        with torch.no_grad():
            output = model(input_tensor)

        # Get the last token prediction, which corresponds to the prediction for the next token
        next_token_logits = output[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()

        # Convert token ID back to a string
        next_token = tokenizer.decode([next_token_id])
        generated_text += next_token

        # Append token ID to our input for the next round
        encoded_input.append(next_token_id)

    return generated_text

seed = "Once upon a"
eval_tokenizer = Tokenizer()
print(generate_text(m, evar, seed))
# %%
