"""
Inference-only code for the BiLSTM Siamese MCQ ranker.

Imports torch, so keep this out of the top level of the Modal script — it is
imported inside the container, where torch is installed.
"""

import json
import pickle
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

N_OPTIONS = 5
PAD_ID, UNK_ID = 0, 1
IDX_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


# ── text encoding, identical to the notebook ──────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode(text, word2id, max_len):
    ids = [word2id.get(w, UNK_ID) for w in clean_text(text).split()][:max_len]
    if not ids:
        ids = [UNK_ID]
    length = len(ids)
    return ids + [PAD_ID] * (max_len - length), length


# ── model, copied verbatim from the notebook ──────────────────────────────────
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, token_ids, lengths):
        emb = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.dropout(out)


class BiLSTMSiamese(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.encoder = LSTMEncoder(
            vocab_size, cfg["embed_dim"], cfg["hidden_dim"],
            cfg["num_layers"], cfg["dropout"],
        )
        enc_dim = self.encoder.output_dim
        self.scorer = nn.Sequential(
            nn.Linear(3 * enc_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(128, 1),
        )

    def forward(self, prompt_ids, prompt_len, option_ids, option_lens):
        p_repr = self.encoder(prompt_ids, prompt_len)
        logits = []
        for i in range(N_OPTIONS):
            o_repr = self.encoder(option_ids[:, i, :], option_lens[:, i])
            inter = torch.cat([p_repr, o_repr, p_repr - o_repr], dim=-1)
            logits.append(self.scorer(inter))
        return torch.cat(logits, dim=-1)


class Solver:
    """Loads the exported artifacts once and answers questions."""

    def __init__(self, artifacts_dir):
        d = Path(artifacts_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        with open(d / "vocab.pkl", "rb") as f:
            self.word2id = pickle.load(f)

        self.model = BiLSTMSiamese(self.cfg["vocab_size"], self.cfg)
        state = torch.load(d / "model_weights.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()
        torch.set_num_threads(1)

    @torch.no_grad()
    def solve(self, prompt, options):
        """options is a list of 5 strings. Returns (top3_string, {label: prob})."""
        p_ids, p_len = encode(prompt, self.word2id, self.cfg["max_len_prompt"])
        enc = [encode(o, self.word2id, self.cfg["max_len_option"]) for o in options]

        logits = self.model(
            torch.tensor([p_ids]),
            torch.tensor([p_len]),
            torch.tensor([[ids for ids, _ in enc]]),
            torch.tensor([[ln for _, ln in enc]]),
        )
        probs = torch.softmax(logits, dim=-1)[0]
        top3 = torch.argsort(probs, descending=True)[:3].tolist()

        prediction = " ".join(IDX_TO_LABEL[i] for i in top3)
        confidences = {
            f"{IDX_TO_LABEL[i]}: {str(options[i])[:60] or '(empty)'}": float(probs[i])
            for i in range(N_OPTIONS)
        }
        return prediction, confidences
