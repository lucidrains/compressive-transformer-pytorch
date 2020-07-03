## Compressive Transformer in Pytorch

Pytorch implementation of <a href="https://openreview.net/forum?id=SylKikSYDH">Compressive Transformers</a>, a variant of Transformer-XL with compressed memory for long-range language modelling. I will also combine this with an idea from <a href="https://arxiv.org/abs/1910.06764">another paper</a> that adds gating at the residual intersection. The memory and the gating may be synergistic, and lead to further improvements in both language modeling as well as reinforcement learning.

## Install

```bash
$ pip install compressive_transformer_pytorch
```

## Usage

```python
import torch
from compressive_transformer_pytorch import CompressiveTransformer

model = CompressiveTransformer(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    seq_len = 1024,
    mem_len = 1024,                # memory length
    cmem_len = 1024 // 4,          # compressed memory buffer length
    cmem_ratio = 4,                # compressed memory ratio, 4 was recommended in paper
    reconstruction_loss_weight = 1,# weight to place on compressed memory reconstruction loss
    attn_dropout = 0.1,            # dropout post-attention
    ff_dropout = 0.1,              # dropout in feedforward
    attn_layer_dropout = 0.1,      # dropout for attention layer output
    gru_gated_residual = True,     # whether to gate the residual intersection, from 'Stabilizing Transformer for RL' paper
    memory_layers = range(6, 13)   # specify which layers to use long-range memory, from 'Do Transformers Need LR Memory' paper
)

inputs = torch.randint(0, 256, (1, 2048))
masks = torch.ones_like(inputs).bool()

segments = inputs.reshape(1, -1, 1024).transpose(0, 1)
masks = masks.reshape(1, -1, 1024).transpose(0, 1)

logits, memories, aux_loss = model(segments[0], mask = masks[0])
logits,        _, aux_loss = model(segments[1], mask = masks[1], memories = memories)

# memories is a named tuple that contains the memory (mem) and the compressed memory (cmem)
```

When training, you can use the `AutoregressiveWrapper` to have memory management across segments taken care of for you. As easy as it gets.

```python
import torch
from compressive_transformer_pytorch import CompressiveTransformer
from compressive_transformer_pytorch import AutoregressiveWrapper

model = CompressiveTransformer(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 1024,
    mem_len = 1024,
    cmem_len = 256,
    cmem_ratio = 4,
    memory_layers = [5,6]
).cuda()

model = AutoregressiveWrapper(model)

inputs = torch.randint(0, 20000, (1, 2048 + 1)).cuda()

for loss, aux_loss in model(inputs, return_loss = True):
    (loss + aux_loss).backward()
    # optimizer step and zero grad
```


## Citations

```bibtex
@misc{rae2019compressive,
    title={Compressive Transformers for Long-Range Sequence Modelling},
    author={Jack W. Rae and Anna Potapenko and Siddhant M. Jayakumar and Timothy P. Lillicrap},
    year={2019},
    eprint={1911.05507},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

```bibtex
@misc{parisotto2019stabilizing,
    title={Stabilizing Transformers for Reinforcement Learning},
    author={Emilio Parisotto and H. Francis Song and Jack W. Rae and Razvan Pascanu and Caglar Gulcehre and Siddhant M. Jayakumar and Max Jaderberg and Raphael Lopez Kaufman and Aidan Clark and Seb Noury and Matthew M. Botvinick and Nicolas Heess and Raia Hadsell},
    year={2019},
    eprint={1910.06764},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

```bibtex
@inproceedings{rae-razavi-2020-transformers,
    title = "Do Transformers Need Deep Long-Range Memory?",
    author = "Rae, Jack  and
      Razavi, Ali",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.672"
}
```
