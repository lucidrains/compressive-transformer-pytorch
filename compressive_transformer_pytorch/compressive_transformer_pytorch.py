import torch
from torch import nn
import torch.nn.functional as F

# helper functions

def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]

# helper classes

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

# rel positional embedding

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, dim, heads, length):
        super().__init__()
        self.scale = dim ** -0.5
        self.weights = nn.Parameter(torch.zeros(length, heads, dim))

    def forward(self, q):
        emb = torch.einsum('bhid,jhd->bhij', q, self.weights.type(q.dtype)) * self.scale
        return shift(emb)

# feedforward

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

# attention.

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
        q, k, v = map(lambda x: x.reshape(b, t, h, -1).transpose(1, 2), (q, k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask = torch.ones(t, t).triu_().bool()
        dots.masked_fill_(mask[None, None, ...], float('-inf'))

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.to_out(out)

# transformer

class CompressiveTransformer(nn.Module):
    def __init__(self, num_tokens, dim, max_seq_len, depth, heads = 8):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = RelativePositionalEmbedding
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
