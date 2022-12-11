import numpy as np

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

# from .transcribe import transcribe as transcribe_function
# from .decoding import detect_langage as detect_langage_function, decode as decode_function 

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

def embed_position(length, dims, max_scale=10000):
    assert dims % 2 == 0 
    scaled_time = torch.arange(length)[:, np.newaxis] * torch.exp(-np.log(max_scale) / (dims // 2 - 1) * torch.arange(dims // 2))[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def qkv_attention(self, q:Tensor, k:Tensor, v:Tensor, mask:Optional[Tensor] = None):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None: qk += mask[:n_ctx, :n_ctx]
        return (F.softmax(qk.float(), dim=-1).to(q.dtype) @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

    def forward(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None, kv_cache:Optional[dict]=None):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        return self.out(self.qkv_attention(q, k, v, mask))





