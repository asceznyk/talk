import numpy as np

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

#
#

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
    scaled_time = torch.arange(length)[:, np.newaxis] * torch.exp(-np.log(max_scale) / (dims // 2 - 1) * torch.arange(dims))[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)



