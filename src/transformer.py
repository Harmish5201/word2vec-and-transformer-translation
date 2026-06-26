import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def read_lines(path, limit=6000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Translation file not found at {path}. Please check data path.")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f][:limit]

def tokenize(text):
    return text.lower().split()

def build_vocab(lines):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for line in lines:
        for tok in tokenize(line):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def encode(text, vocab, max_len=40):
    tokens = ["<bos>"] + tokenize(text)[:max_len] + ["<eos>"]
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, t = self.data[idx]
        return torch.tensor(encode(s, self.src_vocab)), torch.tensor(encode(t, self.tgt_vocab))

class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src, tgt = zip(*batch)
        src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=self.pad_idx)
        tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=self.pad_idx)
        return src, tgt

class SimpleTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, dim=128, heads=4):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, dim)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, dim)
        self.transformer = nn.Transformer(
            d_model=dim, 
            nhead=heads, 
            num_encoder_layers=2, 
            num_decoder_layers=2, 
            batch_first=True
        )
        self.out = nn.Linear(dim, tgt_vocab_size)
        self.pos = nn.Parameter(torch.randn(500, dim))  # max length 500

    def forward(self, src, tgt):
        src = self.src_embed(src) + self.pos[:src.size(1)]
        tgt = self.tgt_embed(tgt) + self.pos[:tgt.size(1)]
        return self.out(self.transformer(src, tgt))

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        inp = tgt[:, :-1]
        target = tgt[:, 1:]
        
        out = model(src, inp)
        loss = loss_fn(out.reshape(-1, out.size(-1)), target.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(dataloader)

def greedy_decode(model, src_tensor, inv_tgt_vocab, device, max_len=20, start=2):
    model.eval()
    src_tensor = src_tensor.unsqueeze(0).to(device)
    src_emb = model.src_embed(src_tensor) + model.pos[:src_tensor.size(1)]
    memory = model.transformer.encoder(src_emb)
    ys = torch.tensor([[start]], device=device)
    
    for _ in range(max_len):
        tgt_emb = model.tgt_embed(ys) + model.pos[:ys.size(1)]
        out = model.transformer.decoder(tgt_emb, memory)
        logits = model.out(out[:, -1, :])
        next_word = logits.argmax(1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == 3:  # <eos>
            break
    return [inv_tgt_vocab.get(t.item(), "<unk>") for t in ys[0, 1:]]