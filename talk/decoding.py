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

@torch.no_grad()
def detect_language(model:"Whisper", mel:Tensor, tokenizer:Tokenizer = None) -> Tuple[Tensor, List[dict]]:
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError("This model dosen't have language tokens")

    single = mel.ndim == 2
    if single: mel = mel.unsqueeze(0)

    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(model.device)
    logits = model.logits(x, mel)[:, 0] ## grab the first from sequence

    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens,  tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs

class Inference:
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_cache()

        if tokens.shape[-1] > self.initial_token_length: tokens = tokens[:, -1:]
        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        for module, tensor in self.kv_cache.items():
            self.kv_cache[module] = tensor[source_indices].detach()

@torch.no_grad()
def decode(model:"Whisper", mel:Tensor, options:DecodingOptions = DecodingOptions()) -> Union[DecodingResult, List[DecodingResult]]: 
    def verify_options():
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return True

    def get_audio_features():
        if options.fp16: mel = mel.half()
        if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
            audio_features = model.encoder(mel)

        if audio_features.dtype != (torch.float16 if options.fp16 else torch.float32):
            raise TypeError(f"audio_features have incorrect dtype: {audio_features.dtype}")
        return audio_features

    def get_initial_tokens() -> Tuple[int]:
        tokens = list(sot_sequence)
        prefix =  options.prefix
        prompt = options.prompt

        if prefix: 
            prefix_tokens = (
                tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
            )

            if sample_len is not None:
                prefix_tokens = prefix_tokens[-(n_ctx//2 - sample_len):]
            tokens = tokens + prefix_tokens

        if prompt: 
            prompt_tokens = (
                tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt            
            )
            tokens = [tokenizer.sot_prev] + prompt_tokens[-(n_ctx//2 - 1):] + tokens

        return tuple(tokens)

    def run():
        audio_features:Tensor = get_audio_features()

    single = mel.ndim == 2
    if single: mel = mel.unsqueeze(0)

    assert verify_options()

    language = options.language or "en"
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=options.task)
    n_group: int = options.beam_size or options.best_of or 1
    n_ctx: int = model.dims.n_text_ctx
    sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

    sot_sequence: Tuple[int] = tokenizer.sot_sequence
    if options.without_timestamps:
        sot_sequence = tokenizer.sot_sequence_including_notimestamps

    initial_tokens: Tuple[int] = get_initial_tokens()
    sample_begin: int = len(initial_tokens)
    sot_index: int = initial_tokens.index(tokenizer.sot)

    inference = Inference(model, len(initial_tokens))
    
    '''sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

    if options.beam_size is not None:
        decoder = BeamSearchDecoder(
            options.beam_size, tokenizer.eot, inference, options.patience
        )
    else:
        decoder = GreedyDecoder(options.temperature, tokenizer.eot)

    logit_filters = []
    if options.suppress_blank:
        logit_filters.append(SuppressBlank(tokenizer, sample_begin))
    if options.suppress_tokens:
        logit_filters.append(SuppressTokens(get_suppress_tokens()))
    if not options.without_timestamps:
        precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
        max_initial_timestamp_index = None
        if options.max_initial_timestamp:
            max_initial_timestamp_index = round(options.max_initial_timestamp / precision)
        logit_filters.append(
            ApplyTimestampRules(tokenizer, sample_begin, max_initial_timestamp_index)
        )

    result = run()'''


    result = [101] ## just for gags!
    if single: result = result[0] 
    return result 




