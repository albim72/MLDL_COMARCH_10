"""
Prosty przykład działania Transformera w PyTorch.
Autor: Marcin Albiniak (wersja edukacyjna)
Cel: Pokazanie idei "Self-Attention" i bloku transformera dla studentów.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Prosty model Self-Attention ----------
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # x: [batch, seq_len, embed_size]
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        # obliczamy uwagi (attention scores)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.embed_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # wynikowy wektor ważony
        out = torch.matmul(attention_weights, values)
        return out

# ---------- Prosty blok Transformera ----------
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.attention = SelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )

    def forward(self, x):
        # warstwa uwagi + normalizacja
        attention = self.attention(x)
        x = self.norm1(x + attention)
        # feed-forward + normalizacja
        forward = self.feed_forward(x)
        out = self.norm2(x + forward)
        return out

# ---------- Przykład użycia ----------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Załóżmy, że mamy 3 "słowa", każde reprezentowane przez 5 liczb (embedding size = 5)
    example_input = torch.rand((1, 3, 5))  # batch=1, seq_len=3, embed_size=5
    print("Wejście:\n", example_input)

    transformer = TransformerBlock(embed_size=5, hidden_size=8)
    output = transformer(example_input)

    print("\nWynik po przejściu przez Transformer Block:\n", output)
