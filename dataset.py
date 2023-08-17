#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#%%
import pyarrow.parquet as pq
import sentencepiece as spm
import string
#%% 
#%%
torch.manual_seed(42)

# DataPrep
# converting parquet into pandas
data = pq.read_table('./parquet-train-00000-of-00004.parquet')
df = data.to_pandas()

#Process text to remove punctuation and make all text lowercase
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["processed_text"] = df["text"].apply(preprocess_text)

#convert data frame to list
text_list = df['processed_text'][:888].tolist()

#create a text file using the list
corpus = "\n".join(text_list)
with open('corpus.txt', 'w') as f:
    f.write(corpus)

#%%
# create sentence piece token model and vocab
spm.SentencePieceTrainer.train(
    input='corpus.txt', 
    model_prefix='tiny_tokenizer', 
    vocab_size=8000,
    model_type='bpe' # or 'unigram'
)
#%%
# load tokens and store in tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load('tiny_tokenizer.model')

#%%
class Tokenizer:
    def __init__(self):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load('tiny_tokenizer.model')
        self.vocab_size = self.tokenizer.get_piece_size()
        f.close()
    
    def encode(self, story):
        return self.tokenizer.encode(story, out_type=int)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, out_type=str)

#%%
class StoryDataset(torch.utils.data.Dataset):
    def __init__(self):
        f = open('corpus.txt', 'r')
        self.story_list = f.read().splitlines()
        self.tokenizer = Tokenizer()
        
    def __len__(self):
        return len(self.story_list)
    
    def __getitem__(self, index):
        story = self.story_list[index]
        tokenized_story = self.tokenizer.encode(story)
        source = tokenized_story[:-1]
        target = tokenized_story[1:]
        return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
    def getMaxLength(self):
        return max(len(story) for story in self.story_list)
         
    def getVocabSize(self):
        return self.tokenizer.vocab_size
# %%