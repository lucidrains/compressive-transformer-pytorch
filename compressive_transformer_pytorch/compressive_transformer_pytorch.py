from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

# helper functions

def to(t):
    return {'dtype': t.dtype, 'device': t.device}

def cast_tuple(el):
    return el if isinstance(el, tuple) else tuple(el)

def default(x, val):
    return val if x is None else x

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
    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out = cast_tuple(out)
        ret = (out[0] + x), *out[1:]
        return ret

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# rel positional embedding

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, dim, heads, length):
        super().__init__()
        self.scale = dim ** -0.5
        self.weights = nn.Parameter(torch.zeros(length, heads, dim))

    def forward(self, q, mem_len = 0):
        seq_len = q.shape[2] + mem_len
        weights = self.weights[:seq_len].type(q.dtype)
        emb = torch.einsum('bhid,jhd->bhij', q, weights) * self.scale
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
    def forward(self, x, **kwargs):
        return self.net(x)

# attention.

SelfAttentionOutput = namedtuple('SelfAttentionOutput', ['out', 'mem'])

class SelfAttention(nn.Module):
    def __init__(self, dim, seq_len, mem_len, heads = 8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.scale = self.dim_head ** (-0.5)

        self.rel_pos_emb = RelativePositionalEmbedding(self.dim_head, heads, seq_len + mem_len)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mem = None, **kwargs):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head
        mem = default(mem, torch.empty(b, 0, e))
        mem_len = mem.shape[1]

        q = self.to_q(x)

        kv_input = torch.cat((mem, x), dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda x: x.reshape(b, -1, h, dim_h).transpose(1, 2), (q, k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        pos_attn = self.rel_pos_emb(q, mem_len)
        dots = dots + pos_attn

        mask = torch.ones(t, t + mem_len).triu_(diagonal = 1 + mem_len).bool()
        dots.masked_fill_(mask[None, None, ...], float('-inf'))

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)

        # calculate next memory - compressed memory calculations will be put here
        mem_next = mem
        if self.seq_len == t:
            mem_next = torch.cat((mem, x), dim=1)[:, -self.mem_len:, :].detach()

        return SelfAttentionOutput(out = self.to_out(out), mem = mem_next)

# transformer

TransformerOutput = namedtuple('TransformerOutput', ['out', 'mem'])

class CompressiveTransformer(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, mem_len = None, heads = 8):
        super().__init__()
        mem_len = default(mem_len, seq_len)
        self.mem_len = mem_len
        self.seq_len = seq_len
        self.depth = depth
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.to_logits = nn.Linear(dim, num_tokens)

        self.attn_layers = nn.ModuleList([Residual(PreNorm(dim, SelfAttention(dim, seq_len, mem_len, heads))) for _ in range(depth)])
        self.ff_layers = nn.ModuleList([Residual(PreNorm(dim, FeedForward(dim))) for _ in range(depth)])

    def forward(self, x, mem = None):
        x = self.token_emb(x)
        b, t, d = x.shape

        mem = default(mem, torch.empty(self.depth, b, 0, d))

        next_mem = []
        for attn, ff, m in zip(self.attn_layers, self.ff_layers, mem):
            x, mem_out = attn(x, mem = m)
            x, = ff(x)
            next_mem.append(mem_out)

        out = self.to_logits(x)
        next_mem = torch.stack(next_mem)

        return TransformerOutput(out = out, mem = next_mem)
