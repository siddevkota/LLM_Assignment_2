import torch
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import nltk

nltk.download("punkt")
from nltk.translate.bleu_score import corpus_bleu
from scratch_transformer import (
    Transformer,
    create_src_mask,
    create_tgt_mask,
    get_tokenizers,
    OpusBooksDataset,
)

# --- Load tokenizers and model ---
src_tokenizer, tgt_tokenizer = get_tokenizers()
src_pad_idx = src_tokenizer.pad_token_id
tgt_pad_idx = tgt_tokenizer.pad_token_id

max_len = 32
d_model = 128
num_heads = 8
d_ff = 64
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_len=max_len,
).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# --- Recreate validation DataLoader ---
dataset = load_dataset("opus_books", "en-fr", split="train")
src_texts = [x["translation"]["en"] for x in dataset]
tgt_texts = [x["translation"]["fr"] for x in dataset]
_, src_val, _, tgt_val = train_test_split(
    src_texts, tgt_texts, test_size=0.2, random_state=42
)
val_data = OpusBooksDataset(src_val, tgt_val, src_tokenizer, tgt_tokenizer, max_len)
val_loader = DataLoader(val_data, batch_size=32)


# --- Token-level accuracy ---
def token_level_accuracy(
    model, data_loader, src_tokenizer, tgt_tokenizer, max_len=32, device="cuda"
):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in data_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_mask = create_src_mask(src, src_tokenizer.pad_token_id)
            tgt_mask = create_tgt_mask(tgt_in, tgt_tokenizer.pad_token_id)
            output = model(src, tgt_in, src_mask, tgt_mask)
            pred = output.argmax(dim=-1)
            mask = tgt_out != tgt_tokenizer.pad_token_id
            correct += ((pred == tgt_out) & mask).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0


# --- Sequence-level accuracy ---
def sequence_level_accuracy(
    model, data_loader, src_tokenizer, tgt_tokenizer, max_len=32, device="cuda"
):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in data_loader:
            src = src.to(device)
            for i in range(src.size(0)):
                src_sentence = src[i].unsqueeze(0)
                ref = tgt_out[i].cpu().tolist()
                # Greedy decode
                pred = [tgt_tokenizer.cls_token_id]
                for _ in range(max_len - 1):
                    tgt = torch.tensor([pred], device=device)
                    src_mask = create_src_mask(src_sentence, src_tokenizer.pad_token_id)
                    tgt_mask = create_tgt_mask(tgt, tgt_tokenizer.pad_token_id)
                    out = model(src_sentence, tgt, src_mask, tgt_mask)
                    next_token = out[0, -1].argmax().item()
                    pred.append(next_token)
                    if next_token == tgt_tokenizer.sep_token_id:
                        break
                # Compare prediction and reference (ignoring padding)
                pred_trim = [t for t in pred[1:] if t != tgt_tokenizer.pad_token_id]
                ref_trim = [t for t in ref if t != tgt_tokenizer.pad_token_id]
                if pred_trim == ref_trim:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0


# --- BLEU score ---
def compute_bleu(
    model, data_loader, src_tokenizer, tgt_tokenizer, max_len=32, device="cuda"
):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for src, tgt_in, tgt_out in data_loader:
            src = src.to(device)
            for i in range(src.size(0)):
                src_sentence = src[i].unsqueeze(0)
                ref = tgt_out[i].cpu().tolist()
                # Greedy decode
                pred = [tgt_tokenizer.cls_token_id]
                for _ in range(max_len - 1):
                    tgt = torch.tensor([pred], device=device)
                    src_mask = create_src_mask(src_sentence, src_tokenizer.pad_token_id)
                    tgt_mask = create_tgt_mask(tgt, tgt_tokenizer.pad_token_id)
                    out = model(src_sentence, tgt, src_mask, tgt_mask)
                    next_token = out[0, -1].argmax().item()
                    pred.append(next_token)
                    if next_token == tgt_tokenizer.sep_token_id:
                        break
                # Convert to tokens
                pred_tokens = tgt_tokenizer.convert_ids_to_tokens(pred[1:-1])
                ref_tokens = tgt_tokenizer.convert_ids_to_tokens(
                    [t for t in ref if t != tgt_tokenizer.pad_token_id]
                )
                hypotheses.append(pred_tokens)
                references.append([ref_tokens])
    return corpus_bleu(references, hypotheses)


# --- Run evaluation ---
print("Evaluating model on validation set...")
acc = token_level_accuracy(
    model, val_loader, src_tokenizer, tgt_tokenizer, max_len=max_len, device=device
)
print(f"Token-level accuracy: {acc:.4f}")

seq_acc = sequence_level_accuracy(
    model, val_loader, src_tokenizer, tgt_tokenizer, max_len=max_len, device=device
)
print(f"Sequence-level accuracy: {seq_acc:.4f}")

bleu = compute_bleu(
    model, val_loader, src_tokenizer, tgt_tokenizer, max_len=max_len, device=device
)
print(f"BLEU score: {bleu:.4f}")


import json

results = {
    "token_level_accuracy": acc,
    "sequence_level_accuracy": seq_acc,
    "bleu": bleu,
}
with open("eval_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved evaluation metrics to eval_metrics.json")
