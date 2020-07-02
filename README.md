## Compressive Transformer in Pytorch (wip)

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
    depth = 6,
    seq_len = 1024,
    mem_len = 1024,               # memory length
    cmem_len = 1024 // 4,         # compressed memory buffer length
    cmem_ratio = 4,               # compressed memory ratio, 4 was recommended in paper
    gru_gated_residual = True     # whether to gate the residual intersection, from 'Stabilizing Transformer for RL' paper
)

inputs = torch.randint(0, 256, (1, 2048))

segments = inputs.reshape(1, -1, 1024).transpose(0, 1)

logits, memories, aux_loss = model(segments[0])
logits,        _, aux_loss = model(segments[1], memories = memories)
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
