import pdb

import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union, Callable, TYPE_CHECKING

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING: from .model import Whisper

class Inference:
    def __init__(self, model:"Whisper", initial_token_length:int, enable_cache:bool):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {} if enable_cache else None
        self.hooks = []

    def logits(self, tokens:Tensor, audio_features:Tensor, log_tensors:bool=False) -> Tensor:
        #if not self.kv_cache:
        #    self.kv_cache, self.hooks = self.model.install_cache()

        #if tokens.shape[-1] > self.initial_token_length: 
        #    tokens = tokens[:, -1:]

        try:
            return self.model.decoder(tokens, audio_features, kv_cache=None, log_tensors=log_tensors)
        except:
            print(f"input tokens: {tokens}") 

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        for module, tensor in self.kv_cache.items():
            self.kv_cache[module] = tensor[source_indices].detach()

class LogitFilter:
    def apply(self, logits:Tensor, tokens:Tensor) -> None:
        raise NotImplementedError

class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer:Tokenizer, sample_begin:int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits:Tensor, tokens:Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf

class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits:Tensor, tokens:Tensor):
        logits[:, self.suppress_tokens] = -np.inf

class ApplyTimestampRules(LogitFilter):
    def __init__(
        self, tokenizer:Tokenizer, sample_begin:int, max_initial_timestamp_index:Optional[int]
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits:Tensor, tokens:Tensor):
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        for k in range(tokens.shape[0]):
            seq = [t for t in tokens[k, self.sample_begin :].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                logits[:, last_allowed + 1 :] = -np.inf

        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf

class TokenDecoder:
    def reset(self): return

    def update(self, tokens:Tensor, logits:Tensor, sum_logprobs:Tensor) -> Tuple[Tensor, bool]:
        raise NotImplementedError

    def finalize(self, tokens:Tensor, sum_logprobs:Tensor) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        raise NotImplementedError

class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature, eot):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens:Tensor, logits:Tensor, sum_logprobs:Tensor):
        if self.temperature <= 0: next_tokens = logits.argmax(dim=-1)
        else: next_tokens = Categorical(logits/self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        sum_logprobs += logprobs[torch.arange(logprobs.shape[0]), next_tokens] * (tokens[:, -1] != self.eot)
        next_tokens[tokens[:, -1] == self.eot] = self.eot
        return torch.cat([tokens, next_tokens[:, None]], dim=-1), (tokens[:, -1] == self.eot).all()

    def finalize(self, tokens:Tensor, sum_logprobs:Tensor):
        return F.pad(tokens, (0,1), value=self.eot), sum_logprobs.tolist()

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

def maximum_likelyhood_ranker(tokens:List[List[Tensor]], sum_logprobs:List[List[float]], length_penalty:Optional[float]):
    def scores(logprobs, lengths):
        result = []
        for logprob, length in zip(logprobs, lengths):
            if length_penalty is None: penalty = length
            else: penalty = ((5 + length) / 6) ** length_penalty
            result.append(logprob/penalty)
        return result

    lengths = [[len(t) for t in s] for s in tokens]
    return [np.argmax(scores(p,l)) for p, l in zip(sum_logprobs, lengths)]

@dataclass
class DecodingOptions:
    task:str = "transcribe" # whether to perform X->X "transcribe" or X->English "translate"
    language:Optional[str] = None # language that the audio is in; uses detected language if None

    # sampling-related options
    temperature:float = 0.0
    sample_len:Optional[int] = None # maximum number of tokens to sample
    best_of:Optional[int] = None # number of independent samples to collect, when t > 0
    beam_size:Optional[int] = None # number of beams in beam search, when t == 0
    patience:Optional[float] = None # patience in beam search (https://arxiv.org/abs/2204.05424)

    # options for ranking generations (either beams or best-of-N samples)
    length_penalty:Optional[float] = None   # "alpha" in Google NMT, None defaults to length norm
    sequence_ranker:Callable = maximum_likelyhood_ranker

    # prompt, prefix, and token suppression
    prompt:Optional[Union[str, List[int]]] = None 
    prefix:Optional[Union[str, List[int]]] = None 
    suppress_blank:bool = True 
    suppress_tokens:Optional[Union[str, Iterable[int]]] = "-1"
    without_timestamps:bool = False
    max_initial_timestamp:Optional[float] = 1.0 
    fp16:bool = False
    log_tensors:bool = False
    enable_cache:bool = True

@dataclass(frozen=True)
class DecodingResult:
    audio_features:Tensor
    language:str
    language_probs:Optional[Dict[str, float]] = None
    tokens:List[int] = field(default_factory=list)
    text:str = ""
    avg_logprob:float = np.nan
    no_speech_prob:float = np.nan
    temperature:float = np.nan
    compression_ratio:float = np.nan
    
@torch.no_grad()
def decode(model:"Whisper", mel:Tensor, options:DecodingOptions = DecodingOptions()) -> Union[DecodingResult, List[DecodingResult]]:
    print(f"decoding, task = {options.task}..")

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

    def get_audio_features(mel:Tensor) -> Tensor:
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
    
    def get_suppress_tokens() -> Tuple[int]:
        suppress_tokens = options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm]
        )
        if tokenizer.no_speech is not None: 
            suppress_tokens.append(tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def get_language(audio_features:Tensor, tokens:Tensor):
        languages = [options.language] * audio_features.shape[0]
        lang_probs = None

        if options.task == "lang_id" or options.language is None:
            lang_tokens, lang_probs = model.detect_language(audio_features, tokenizer)
            languages = [max(p, key=p.get) for p in lang_probs]
            if options.language is None:
                tokens[:, sot_index + 1] = lang_tokens

        return languages, lang_probs

    def main_loop(audio_features:Tensor, tokens:Tensor):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = audio_features.shape[0]
        sum_logprobs = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(sample_len):
                logits = inference.logits(tokens, audio_features, options.log_tensors)

                if i == 0 and tokenizer.no_speech is not None:
                    probs_at_sot = logits[:, sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, tokenizer.no_speech].tolist()

                logits = logits[:, -1] ## consider the last token only

                for logit_filter in logit_filters:
                    logit_filter.apply(logits, tokens)

                tokens, completed = decoder.update(tokens, logits, sum_logprobs)
                if tokens.shape[-1] > n_ctx or completed:
                    break
        finally: 
            inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    def run(mel:Tensor) -> List[DecodingResult]:
        decoder.reset()
        n_audio:int = mel.shape[0]
        audio_features:Tensor = get_audio_features(mel)
        tokens:Tensor = torch.tensor([initial_tokens]).repeat(n_audio, 1)

        languages, lang_probs = get_language(audio_features, tokens)
        if options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                ) for features, language, probs in zip(audio_features, languages, lang_probs)
            ]

        audio_features = audio_features.repeat_interleave(n_group, dim=0)
        tokens = tokens.repeat_interleave(n_group, dim=0)
        print([tokenizer.decode(t) for t in tokens])
        tokens, sum_logprobs, no_speech_probs = main_loop(audio_features, tokens)

        audio_features = audio_features[::n_group]
        no_speech_probs = no_speech_probs[::n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, n_group, -1)
        sum_logprobs = Sum_logprobs.reshape(n_audio, n_group)
        tokens, sum_logprobs = decoder.finalize(tokens, sum_logprobs)

        tokens:List[List[Tensor]] = [
            [t[sample_begin:(t==tokenizer.eot).nonzero()[0,0]] for t in s] for s in tokens
        ]

        selected = sequence_ranker(tokens, sum_logprobs, options.length_penalty)
        tokens:List[List[int]] = [s[i].tolist() for i, s in zip(selected, tokens)] 
        texts:List[str] = [tokenizer.decode(s).strip() for s in tokens]
        sum_logprobs:List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs:List[float] = [lp/(len(s)+1) for s, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=options.temperature,
                compression_ratio=compression_ratio(text)
            )
            for (text, language, tokens, features, avg_logprob, no_speech_prob) in zip(*fields)
        ]

    single = mel.ndim == 2
    if single: mel = mel.unsqueeze(0)

    assert verify_options()

    language = options.language or "en"
    tokenizer:Tokenizer = get_tokenizer(model.is_multilingual, language=language, task=options.task)
    n_group:int = options.beam_size or options.best_of or 1
    n_ctx:int = model.dims.n_text_ctx
    sample_len:int = options.sample_len or model.dims.n_text_ctx // 2

    sot_sequence:Tuple[int] = tokenizer.sot_sequence
    if options.without_timestamps:
        sot_sequence = tokenizer.sot_sequence_including_notimestamps

    initial_tokens:Tuple[int] = get_initial_tokens()
    sample_begin:int = len(initial_tokens)
    sot_index:int = initial_tokens.index(tokenizer.sot)

    inference = Inference(model, len(initial_tokens), options.enable_cache)

    sequence_ranker = options.sequence_ranker

    '''if options.beam_size is not None:
        decoder = BeamSearchDecoder(options.beam_size, tokenizer.eot, inference, options.patience)
    else:''' 
        
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

    result = run(mel)
    if single: result = result[0]

    return result 





