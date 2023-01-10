import numpy as np

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from torch.nn import LayerNorm, Linear, Conv1d

from .decoding import detect_language as detect_language_function, decode as decode_function 

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
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def qkv_attention(self, q:Tensor, k:Tensor, v:Tensor, mask:Optional[Tensor]=None, log_tensors:bool=False):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if log_tensors:
            print(f"q.shape = {q.shape}")
            print(f"k.shape = {k.shape}")
            print(f"mask.shape = {mask.shape}")

        qk = q @ k
        if mask is not None: qk += mask[:n_ctx, :n_ctx]
        return (F.softmax(qk.float(), dim=-1).to(q.dtype) @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

    def forward(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None, kv_cache:Optional[dict]=None, log_tensors:bool=False):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            inp = x if xa is None else xa 
            k = self.key(inp)
            v = self.value(inp)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        return self.out(self.qkv_attention(q, k, v, mask, log_tensors=log_tensors))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state:int, n_head:int, cross_attention:bool=False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None 

        n_mlp = 4 * n_state
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None, kv_cache:Optional[dict]=None, log_tensors:bool=False):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache, log_tensors=log_tensors)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache, log_tensors=log_tensors)
        return x + self.mlp(self.mlp_ln(x))

class AudioEncoder(nn.Module):
    def __init__(self, n_mels:int, n_ctx:int, n_state:int, n_head:int, n_layer:int):
        super().__init__()

        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        self.register_buffer("positional_embedding", embed_position(n_ctx, n_state))

        self.blocks:Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x:Tensor):
        x = F.gelu(self.conv1(x)) ## x.shape (batch_size, n_mels, n_ctx)
        x = F.gelu(self.conv2(x)) ## x.shape (batch_size, n_state, n_ctx)

        x = x.permute(0, 2, 1) + self.positional_embedding ##positional_embedding is broadcasted in batch_size dim

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)

class TextDecoder(nn.Module):
    def __init__(self, n_vocab:int, n_ctx:int, n_state:int, n_head:int, n_layer:int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.register_buffer("mask", torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1), persistent=False)

        self.blocks:Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )

        self.ln = LayerNorm(n_state)

    def forward(self, x:Tensor, xa:Tensor, kv_cache:Optional[dict]=None, log_tensors:bool=False):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset:offset+x.shape[-1]]

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache, log_tensors=log_tensors)

        return (self.ln(x) @ torch.transpose(self.token_embedding.weight, 0, 1))

class Talk(nn.Module):
    def __init__(self, dims:ModelDimensions):
        super().__init__()
        self.dims = dims

        self.encoder = AudioEncoder(
            dims.n_mels,
            dims.n_audio_ctx,
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer
        )

        self.decoder = TextDecoder(
            dims.n_vocab,
            dims.n_text_ctx,
            dims.n_text_state,
            dims.n_text_head,
            dims.n_text_layer
        )

    @property
    def device(self): return next(self.parameters()).device

    @property
    def is_multilingual(self): return self.dims.n_vocab == 51865

    def embed_audio(self, mel:Tensor): return self.encoder(mel)

    def logits(self, tokens:Tensor, audio_features:Tensor): 
        return self.decoder(tokens, audio_features)

    def forward(self, mel:Tensor, tokens:Tensor):
        return self.decoder(tokens, self.encoder(mel))
    
    def install_cache(self, cache:Optional[dict] = None): 
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer:nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    decode = decode_function



