import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from scratch_transformer import (
    Transformer,
    create_src_mask,
    create_tgt_mask,
    get_tokenizers,
)
import json

train_losses = np.load("train_losses.npy")
val_losses = np.load("val_losses.npy")

max_len = 32
d_model = 128
num_heads = 8
d_ff = 64
num_layers = 2

# Load tokenizers
src_tokenizer, tgt_tokenizer = get_tokenizers()
src_pad_idx = src_tokenizer.pad_token_id
tgt_pad_idx = tgt_tokenizer.pad_token_id

# Load model
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

with open("eval_metrics.json", "r") as f:
    eval_metrics = json.load(f)

st.set_page_config(layout="wide")
st.title("Assignment 2.1")

st.sidebar.header("Model Information")
st.sidebar.write(f"**Model:** Custom Transformer (from scratch)")
st.sidebar.write(f"**Dataset:** opus_books (en-fr)")
st.sidebar.write(f"**Max Length:** {max_len}")
st.sidebar.write(f"**d_model:** {d_model}")
st.sidebar.write(f"**num_heads:** {num_heads}")
st.sidebar.write(f"**d_ff:** {d_ff}")
st.sidebar.write(f"**num_layers:** {num_layers}")
st.sidebar.write(f"**Epochs:** {len(train_losses)}")
st.sidebar.write(f"**Final Train Loss:** {train_losses[-1]:.4f}")
st.sidebar.write(f"**Final Val Loss:** {val_losses[-1]:.4f}")
st.sidebar.write(f"--------------------------------")
st.sidebar.subheader("Evaluation Metrics")
st.sidebar.write(
    f"**Token-level accuracy:** {eval_metrics['token_level_accuracy']:.4f}"
)
st.sidebar.write(
    f"**Sequence-level accuracy:** {eval_metrics['sequence_level_accuracy']:.4f}"
)
st.sidebar.write(f"**BLEU score:** {eval_metrics['bleu']:.4f}")
st.sidebar.write(f"--------------------------------")


st.sidebar.subheader("Train vs Validation Loss Curve")
fig, ax = plt.subplots()
ax.plot(train_losses, label="Train Loss")
ax.plot(val_losses, label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Train vs Validation Loss")
ax.legend()
st.sidebar.pyplot(fig)

st.subheader("Custom Transformer: English to French Translation Demo")


def greedy_decode(model, src_sentence, src_tokenizer, tgt_tokenizer, max_len=32):
    device = next(model.parameters()).device
    src = src_tokenizer(
        src_sentence,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )["input_ids"].to(device)
    src_mask = create_src_mask(src, src_tokenizer.pad_token_id)
    enc_out = model.encode(src, src_mask)
    tgt = torch.tensor([[tgt_tokenizer.cls_token_id]]).to(device)
    for _ in range(max_len - 1):
        tgt_mask = create_tgt_mask(tgt, tgt_tokenizer.pad_token_id)
        dec_out = model.decode(tgt, enc_out, tgt_mask, src_mask)
        logits = model.proj(dec_out)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == tgt_tokenizer.sep_token_id:
            break
    tokens = tgt_tokenizer.convert_ids_to_tokens(tgt[0].tolist())
    return tgt_tokenizer.convert_tokens_to_string(tokens[1:-1])  # skip BOS and EOS


with st.form("translation_form"):
    user_input = st.text_input("Enter English text:")
    submitted = st.form_submit_button("Translate")
    if submitted and user_input.strip():
        with st.spinner("Translating..."):
            translation = greedy_decode(
                model, user_input, src_tokenizer, tgt_tokenizer, max_len=max_len
            )
        st.write(f"**French translation:** {translation}")
    elif submitted:
        st.warning("Please enter some English text.")

st.markdown("---")
st.markdown(
    "**Tip:** The model is a small transformer trained from scratch for demonstration. For best results, use short, simple sentences."
)
