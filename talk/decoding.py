from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING: from .model import Whisper

@dataclass(frozen=True)
class DecodingOptions:
    task: str = "transcribe" # whether to perform X->X "transcribe" or X->English "translate"
    language: Optional[str] = None # language that the audio is in; uses detected language if None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None # maximum number of tokens to sample
    best_of: Optional[int] = None # number of independent samples to collect, when t > 0
    beam_size: Optional[int] = None # number of beams in beam search, when t == 0
    patience: Optional[float] = None # patience in beam search (https://arxiv.org/abs/2204.05424)

    # options for ranking generations (either beams or best-of-N samples)
    length_penalty: Optional[float] = None   # "alpha" in Google NMT, None defaults to length norm

    # prompt, prefix, and token suppression
    prompt: Optional[Union[str, List[int]]] = None 
    prefix: Optional[Union[str, List[int]]] = None 
    suppress_blank: bool = True 
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    without_timestamps: bool = False
    max_initial_timestamp: Optional[float] = 1.0 
    fp16: bool = True 

@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan




