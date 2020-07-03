from collections import namedtuple

from functools import partial
from inspect import isfunction
import torch
from torch import nn
import torch.nn.functional as F

# structs

Memory = namedtuple('Memory', ['mem', 'compressed_mem'])

# helper functions

def to(t):
    return {'dtype': t.dtype, 'device': t.device}

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

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

def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]

# full attention for calculating auxiliary reconstruction loss
def full_attn(q, k, v):
    dots = torch.einsum('bhid,bhjd->bhij', q, k)
    attn = dots.softmax(dim=-1)
    return torch.einsum('bhij,bhjd->bhid', attn, v)

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

class GRUGating(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRU(dim, dim)

    def forward(self, x, **kwargs):
        batch, dim = x.shape[0], self.dim
        out = self.fn(x, **kwargs)
        (y, *rest) = cast_tuple(out)

        gated_output, _ = self.gru(
            y.reshape(1, -1, dim),
            x.reshape(1, -1, dim)
        )

        gated_output = gated_output.reshape(batch, -1, dim)
        ret = gated_output, *rest
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

# feedforward

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# attention.

class SelfAttention(nn.Module):
    def __init__(self, dim, seq_len, mem_len, cmem_len, cmem_ratio = 4, heads = 8, attn_dropout = 0., dropout = 0.):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.scale = self.dim_head ** (-0.5)

        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memories = None, pos_emb = None, input_mask = None, calc_memory = True, **kwargs):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        mem = cmem = None
        if memories is not None:
            mem, cmem = memories

        mem = default(mem, lambda: torch.empty(b, 0, e, **to(x)))
        cmem = default(cmem, lambda: torch.empty(b, 0, e, **to(x)))

        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]

        q = self.to_q(x)

        kv_input = torch.cat((cmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: x.reshape(b, -1, h, dim_h).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * self.scale
            pos_dots = shift(pos_dots)
            dots = dots + pos_dots

        if input_mask is not None:
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            mask = F.pad(mask, (mem_len + cmem_len, 0), value = False)
            dots.masked_fill_(~mask, mask_value)

        mask = torch.ones(t, kv_len, **to(x)).triu_(diagonal = 1 + kv_len).bool()
        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        logits = self.to_out(out)
        logits = self.dropout(logits)

        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad = True, **to(q))

        if self.seq_len < t or not calc_memory:
            return logits, Memory(mem = new_mem, compressed_mem = new_cmem), aux_loss

        # calculate memory and compressed memory

        old_mem, new_mem = split_at_index(1, -self.mem_len, torch.cat((mem, x), dim=1))
        old_mem_padding = old_mem.shape[1] % self.cmem_ratio

        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value = 0.)

        if old_mem.shape[1] != 0:
            compressed_mem = self.compress_mem_fn(old_mem)
            old_cmem, new_cmem = split_at_index(1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1))

            # calculate auxiliary loss if training

            if self.training:
                cmem_k, cmem_v = self.to_kv(compressed_mem).chunk(2, dim=-1)
                cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))

                old_mem_range = slice(- min(mem_len, self.mem_len) - self.seq_len, -self.seq_len)
                old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))

                q, old_mem_k, old_mem_v, cmem_k, cmem_v = map(lambda x: x.detach(), (q, old_mem_k, old_mem_v, cmem_k, cmem_v))

                aux_loss = F.mse_loss(
                    full_attn(q, old_mem_k, old_mem_v),
                    full_attn(q, cmem_k, cmem_v)
                )


        return logits, Memory(mem = new_mem, compressed_mem = new_cmem), aux_loss

# transformer

class CompressiveTransformer(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, memory_layers = None, mem_len = None, cmem_len = None, cmem_ratio = 4, heads = 8, gru_gated_residual = True, attn_dropout = 0., ff_dropout = 0., attn_layer_dropout = 0., reconstruction_loss_weight = 1.):
        super().__init__()
        mem_len = default(mem_len, seq_len)
        cmem_len = default(cmem_len, mem_len // cmem_ratio)
        memory_layers = default(memory_layers, list(range(1, depth + 1)))

        self.seq_len = seq_len

        self.depth = depth
        self.memory_layers = list(memory_layers)

        self.token_emb = nn.Embedding(num_tokens, dim)

        seq_and_mem_len = seq_len + mem_len + cmem_len
        self.pos_emb = nn.Parameter(torch.zeros(heads, seq_and_mem_len, dim // heads))
        self.to_logits = nn.Linear(dim, num_tokens)

        wrapper = partial(GRUGating, dim) if gru_gated_residual else Residual

        self.attn_layers = nn.ModuleList([wrapper(PreNorm(dim, SelfAttention(dim, seq_len, mem_len, cmem_len, cmem_ratio, heads, dropout = attn_layer_dropout, attn_dropout = attn_dropout))) for _ in range(depth)])
        self.ff_layers = nn.ModuleList([wrapper(PreNorm(dim, FeedForward(dim, dropout = ff_dropout))) for _ in range(depth)])

        self.reconstruction_loss_weight = reconstruction_loss_weight

    def forward(self, x, memories = None, mask = None):
        x = self.token_emb(x)
        b, t, d = x.shape

        mem = cmem = None
        if memories is not None:
            mem, cmem = memories

        num_memory_layers = len(self.memory_layers)
        mem = default(mem, lambda: torch.empty(num_memory_layers, b, 0, d, **to(x)))
        cmem = default(cmem, lambda: torch.empty(num_memory_layers, b, 0, d, **to(x)))

        total_len = mem.shape[2] + cmem.shape[2] + t
        pos_emb = self.pos_emb[:, (self.seq_len - t):total_len]

        next_mem = []
        next_cmem = []
        aux_loss = torch.tensor(0., requires_grad = True, **to(x))

        mem_iter = iterate_tensor(mem)
        cmem_iter = iterate_tensor(cmem)

        for ind, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):
            layer_num = ind + 1
            use_memory = layer_num in self.memory_layers

            memories = None
            if use_memory:
                memories = (next(mem_iter), next(cmem_iter))

            x, (mem_out, cmem_out), layer_aux_loss = attn(x, memories = memories, calc_memory = use_memory, input_mask = mask, pos_emb = pos_emb)
            x, = ff(x)

            if use_memory:
                next_mem.append(mem_out)
                next_cmem.append(cmem_out)

            aux_loss = aux_loss + layer_aux_loss

        out = self.to_logits(x)

        next_mem, next_cmem = map(torch.stack, (next_mem, next_cmem))
        next_mem, next_cmem = map(lambda x: x.detach(), (next_mem, next_cmem))

        aux_loss = aux_loss * self.reconstruction_loss_weight
        return out, Memory(mem = next_mem, compressed_mem = next_cmem), aux_loss
