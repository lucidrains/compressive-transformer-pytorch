from collections import namedtuple

from inspect import isfunction
import torch
from torch import nn
import torch.nn.functional as F

# helper functions

def to(t):
    return {'dtype': t.dtype, 'device': t.device}

def cast_tuple(el):
    return el if isinstance(el, tuple) else tuple(el)

def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

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

class ConvCompress(nn.Module):
    def __init__(self, dim, ratio = 4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride = ratio)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)

# rel positional embedding

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, dim, heads, length):
        super().__init__()
        self.scale = dim ** -0.5
        self.weights = nn.Parameter(torch.zeros(length, heads, dim))

    def forward(self, q, mem_len):
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

SelfAttentionOutput = namedtuple('SelfAttentionOutput', ['out', 'mem', 'cmem', 'aux_loss'])

# full attention for compressed memory auxiliary loss
def full_attn(q, k, v):
    dots = torch.einsum('bhid,bhjd->bhij', q, k)
    attn = dots.softmax(dim=-1)
    return torch.einsum('bhij,bhjd->bhid', attn, v)

class SelfAttention(nn.Module):
    def __init__(self, dim, seq_len, mem_len, cmem_len, cmem_ratio = 4, heads = 8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.scale = self.dim_head ** (-0.5)

        seq_and_mem_len = seq_len + mem_len + cmem_len
        self.rel_pos_emb = RelativePositionalEmbedding(self.dim_head, heads, seq_and_mem_len)
        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mem = None, cmem = None, **kwargs):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        mem = default(mem, lambda: torch.empty(b, 0, e))
        cmem = default(cmem, lambda: torch.empty(b, 0, e))

        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]

        q = self.to_q(x)

        kv_input = torch.cat((cmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: x.reshape(b, -1, h, dim_h).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        pos_attn = self.rel_pos_emb(q, mem_len + cmem_len)
        dots = dots + pos_attn

        mask = torch.ones(t, kv_len).triu_(diagonal = 1 + kv_len).bool()
        dots.masked_fill_(mask[None, None, ...], float('-inf'))

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)

        # calculate next memory - compressed memory calculations will be put here
        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad = True, **to(q))

        if self.seq_len == t:
            old_mem, new_mem = split_at_index(1, -self.mem_len, torch.cat((mem, x), dim=1))
            old_mem_padding = old_mem.shape[1] % self.cmem_ratio

            if old_mem_padding != 0:
                old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value = 0.)

            if old_mem.shape[1] != 0:
                compressed_mem = self.compress_mem_fn(old_mem)
                old_cmem, new_cmem = split_at_index(1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1))

                cmem_k, cmem_v = self.to_kv(compressed_mem).chunk(2, dim=-1)
                cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))

                old_mem_range = slice(- min(mem_len, self.mem_len) - self.seq_len, -self.seq_len)
                old_mem_k, old_mem_v = k[:, :, old_mem_range].clone(), v[:, :, old_mem_range].clone()

                old_mem_attn_output = full_attn(q.detach(), old_mem_k.detach(), old_mem_v.detach())
                compressed_mem_attn_output = full_attn(q.detach(), cmem_k.detach(), cmem_v.detach())
                aux_loss = F.mse_loss(old_mem_attn_output, compressed_mem_attn_output)

        return SelfAttentionOutput(out = self.to_out(out), mem = new_mem.detach(), cmem = new_cmem.detach(), aux_loss = aux_loss)

# transformer

TransformerOutput = namedtuple('TransformerOutput', ['out', 'mem', 'cmem', 'aux_loss'])

class CompressiveTransformer(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, mem_len = None, cmem_len = None, cmem_ratio = 4, heads = 8):
        super().__init__()
        mem_len = default(mem_len, seq_len)
        cmem_len = default(cmem_len, mem_len // cmem_ratio)

        self.depth = depth
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.to_logits = nn.Linear(dim, num_tokens)

        self.attn_layers = nn.ModuleList([Residual(PreNorm(dim, SelfAttention(dim, seq_len, mem_len, cmem_len, cmem_ratio, heads))) for _ in range(depth)])
        self.ff_layers = nn.ModuleList([Residual(PreNorm(dim, FeedForward(dim))) for _ in range(depth)])

    def forward(self, x, mem = None, cmem = None):
        x = self.token_emb(x)
        b, t, d = x.shape

        mem = default(mem, lambda: torch.empty(self.depth, b, 0, d))
        cmem = default(cmem, lambda: torch.empty(self.depth, b, 0, d))

        next_mem = []
        next_cmem = []
        aux_loss = torch.zeros(1, requires_grad = True, **to(x))

        for attn, ff, m, c in zip(self.attn_layers, self.ff_layers, mem, cmem):
            x, mem_out, cmem_out, aux_loss = attn(x, mem = m, cmem = c)
            x, = ff(x)
            next_mem.append(mem_out)
            next_cmem.append(cmem_out)

        out = self.to_logits(x)
        next_mem = torch.stack(next_mem)
        next_cmem = torch.stack(next_cmem)

        return TransformerOutput(out = out, mem = next_mem, cmem = next_cmem, aux_loss = aux_loss)
