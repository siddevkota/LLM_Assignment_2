import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
    def forward(self, x):
        return self.norm(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_weights = None  # For visualization
    def forward(self, q, k, v, mask=None):
        B, S_q, D = q.size()
        S_k = k.size(1)
        q = self.W_q(q).view(B, S_q, self.num_heads, self.d_k).transpose(1,2)  # (B, h, S_q, d_k)
        k = self.W_k(k).view(B, S_k, self.num_heads, self.d_k).transpose(1,2)  # (B, h, S_k, d_k)
        v = self.W_v(v).view(B, S_k, self.num_heads, self.d_k).transpose(1,2)  # (B, h, S_k, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)      # (B, h, S_q, S_k)
        if mask is not None:
            # mask shape should be broadcastable to (B, 1, S_q, S_k)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        self.attn_weights = attn.detach().cpu()
        context = torch.matmul(attn, v)  # (B, h, S_q, d_k)
        context = context.transpose(1,2).contiguous().view(B, S_q, D)
        return self.W_o(context)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttentionBlock(d_model, num_heads)
        self.res1 = ResidualConnection(d_model, dropout)
        self.ff = FeedForwardBlock(d_model, d_ff)
        self.res2 = ResidualConnection(d_model, dropout)
    def forward(self, x, src_mask):
        x = self.res1(x, lambda x: self.attn(x, x, x, src_mask))
        x = self.res2(x, self.ff)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(d_model, num_heads)
        self.res1 = ResidualConnection(d_model, dropout)
        self.cross_attn = MultiHeadAttentionBlock(d_model, num_heads)
        self.res2 = ResidualConnection(d_model, dropout)
        self.ff = FeedForwardBlock(d_model, d_ff)
        self.res3 = ResidualConnection(d_model, dropout)
    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = self.res1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.res2(x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.res3(x, self.ff)
        return x


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return self.proj(x)



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=32, num_heads=2, d_ff=64, num_layers=2, max_len=32):
        super().__init__()
        self.src_embed = InputEmbeddings(src_vocab_size, d_model)
        self.tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)
        self.src_pos = PositionalEncoding(d_model, max_len)
        self.tgt_pos = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.proj = ProjectionLayer(d_model, tgt_vocab_size)
    def encode(self, src, src_mask):
        x = self.src_pos(self.src_embed(src))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x
    def decode(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.tgt_pos(self.tgt_embed(tgt))
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return self.proj(dec_out)


def create_src_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)

def create_tgt_mask(tgt, pad_idx):
    B, S = tgt.size()
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
    subsequent_mask = torch.tril(torch.ones((S, S), device=tgt.device)).bool()  # (S,S)
    return pad_mask & subsequent_mask  # (B,1,S,S)



class OpusBooksDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len=16):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.data = []
        for src, tgt in zip(src_texts, tgt_texts):
            src_enc = src_tokenizer(src, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
            tgt_enc = tgt_tokenizer(tgt, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
            src_ids = src_enc['input_ids'].squeeze(0)
            tgt_ids = tgt_enc['input_ids'].squeeze(0)
            self.data.append((src_ids, tgt_ids))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src = self.data[idx][0]
        tgt = self.data[idx][1]
        return src, tgt[:-1], tgt[1:]  # input, target

def get_tokenizers():
    src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tgt_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    return src_tokenizer, tgt_tokenizer


def train():
    # Tokenizers
    src_tokenizer, tgt_tokenizer = get_tokenizers()
    src_pad_idx = src_tokenizer.pad_token_id
    tgt_pad_idx = tgt_tokenizer.pad_token_id

    # Dataset
    dataset = load_dataset("opus_books", "en-fr", split="train")
    src_texts = [x["translation"]["en"] for x in dataset]
    tgt_texts = [x["translation"]["fr"] for x in dataset]
    src_train, src_val, tgt_train, tgt_val = train_test_split(src_texts, tgt_texts, test_size=0.2, random_state=42)
    max_len = 32
    train_data = OpusBooksDataset(src_train, tgt_train, src_tokenizer, tgt_tokenizer, max_len)
    val_data = OpusBooksDataset(src_val, tgt_val, src_tokenizer, tgt_tokenizer, max_len)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=128, num_heads=8, d_ff=64, num_layers=2, max_len=max_len
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    train_losses, val_losses, attn_weights = [], [], []

    for epoch in range(20):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for src, tgt_in, tgt_out in train_bar:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_mask = create_src_mask(src, src_pad_idx)
            tgt_mask = create_tgt_mask(tgt_in, tgt_pad_idx)
            optimizer.zero_grad()
            output = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        train_losses.append(total_loss/len(train_loader))

        # Save attention weights from the first batch of the last epoch layer for visualization
        attn = model.encoder[0].attn.attn_weights
        if attn is not None:
            attn_weights.append(attn[0,0].cpu().numpy())  # (seq, seq) for first head, first sample

        # Validation
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        with torch.no_grad():
            for src, tgt_in, tgt_out in val_bar:
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
                src_mask = create_src_mask(src, src_pad_idx)
                tgt_mask = create_tgt_mask(tgt_in, tgt_pad_idx)
                output = model(src, tgt_in, src_mask, tgt_mask)
                loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
        val_losses.append(val_loss/len(val_loader))
        print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # Save for Streamlit
    np.save("train_losses.npy", np.array(train_losses))
    np.save("val_losses.npy", np.array(val_losses))
    np.save("attn_weights.npy", np.array(attn_weights))

    # Plot and save loss curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    return model, src_tokenizer, tgt_tokenizer, max_len


def greedy_decode(model, src_sentence, src_tokenizer, tgt_tokenizer, max_len=16):
    device = next(model.parameters()).device
    src = src_tokenizer(src_sentence, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')['input_ids'].to(device)
    src_mask = create_src_mask(src, src_tokenizer.pad_token_id)
    enc_out = model.encode(src, src_mask)
    tgt = torch.tensor([[tgt_tokenizer.cls_token_id]]).to(device)
    for _ in range(max_len-1):
        tgt_mask = create_tgt_mask(tgt, tgt_tokenizer.pad_token_id)
        dec_out = model.decode(tgt, enc_out, tgt_mask, src_mask)
        logits = model.proj(dec_out)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == tgt_tokenizer.sep_token_id:
            break
    tokens = tgt_tokenizer.convert_ids_to_tokens(tgt[0].tolist())
    return tgt_tokenizer.convert_tokens_to_string(tokens[1:-1])  # skip BOS and EOS


if __name__ == "__main__":
    model, src_tokenizer, tgt_tokenizer, max_len = train()
    test_sent = "This is a book."
    print("\nEN:", test_sent)
    print("FR (predicted):", greedy_decode(model, test_sent, src_tokenizer, tgt_tokenizer, max_len=max_len))