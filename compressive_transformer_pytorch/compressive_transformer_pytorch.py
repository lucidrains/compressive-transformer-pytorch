import torch
from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.LeakyReLU(inplace = True),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** (-0.5)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)
    def forward(self, x):
        b, t, e, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, -1, h, e).transpose(1, 2), (q, k, v))
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.to_out(out)

class CompressiveTransformer(nn.Module):
    def __init__(self, num_tokens, dim, max_seq_len, depth, heads = 8):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.to_logits = nn.Linear(dim, num_tokens)

        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, SelfAttention(dim, heads))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.token_emb(x)
        out = self.layers(x)
        out = self.to_logits(x)
        return out
