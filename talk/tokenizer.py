import os
import numpy as np

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch

from transformers import GPT2TokenizerFast

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "iw": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

@dataclass(frozen=True)
class Tokenizer:
    tokenizer: "GPT2TokenizerFast"
    language: Optional[str]
    sot_sequence: Tuple[int]

    def _token_id(self, token): return self.tokenizer.encode(token)[0]

    def encode(self, text:str, **kwargs): return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, ids:Union[int, List[int], np.ndarray, torch.Tensor], **kwargs): return self.tokenizer.decode(ids, **kwargs)

    def decode_with_timestamps(self, token_ids):
        outputs = [[]]
        for token_id in token_ids:
            if token_id >= self.timestamp_begin:
                outputs.append(f"<|{(token_id - self.timestamp_begin) * 0.02:.2f}|>")
                outputs.append([])
            else: outputs[-1].append(token_id)
        
        outputs = [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        return "".join(outputs)

    @property
    @lru_cache()
    def eot(self) -> int:return self.tokenizer.eos_token_id

    @property
    @lru_cache()
    def sot(self) -> int: return self._token_id("<|startoftranscript|>")

    @property
    @lru_cache()
    def sot_lm(self) -> int: return self._token_id("<|startoflm|>")

    @property
    @lru_cache()
    def sot_prev(self) -> int: return self._token_id("<|startofprev|>")

    @property
    @lru_cache()
    def no_speech(self) -> int: return self._token_id("<|nospeech|>")

    @property
    @lru_cache()
    def no_timestamps(self) -> int: return self._token_id("<|notimestamps|>") 

    @property
    @lru_cache()
    def timestamp_begin(self) -> int: return self.tokenizer.all_special_ids[-1]+1

    @property
    @lru_cache()
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @property
    @lru_cache()
    def language_token(self) -> int:
        if self.language is None:
            raise ValueError(f"The Tokenizer doesn't have a language")
        return self.tokenizer.vocab[f"<|{self.language}|>"]

    @property
    @lru_cache()
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in zip(
            self.tokenizer.additional_special_tokens,
            self.tokenizer.additional_special_tokens_ids,
        ):
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    @property
    @lru_cache()
    def all_language_codes(self) -> Tuple[str]: 
        return tuple(self.decode([l]).strip("<|>") for l in self.all_language_tokens)

@lru_cache(maxsize=None)
def build_tokenizer(name: str="gpt2"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    tokenizer = GPT2TokenizerFast.from_pretrained(os.path.join(os.path.dirname(__file__), "assets", name)) 

    tokenizer.add_special_tokens(dict(additional_special_tokens=[
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]))
    return tokenizer

@lru_cache(maxsize=None)
def get_tokenizer(
        multilingual:bool, 
        *, 
        task: Optional[str] = None, 
        language: Optional[str] = None
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language not in TO_LANGUAGE_CODE:
                raise ValueError(f"Unsupported language {language}")
            language = TO_LANGUAGE_CODE[language] 

    task, language = task or "transcribe", language or "en" 
    if not multilingual:
        task, language = None, None

    tokenizer = build_tokenizer(name="gpt2" if not multilingual else "multilingual") 

    all_special_ids = tokenizer.all_special_ids 
    transcribe = all_special_ids[-5] 
    sot_sequence = [all_special_ids[1]]
    if language is not None: sot_sequence.append(tokenizer.vocab[f'<|{language}|>'])
    if task is not None: 
        sot_sequence.append(
            all_special_ids[-5] if task == "transcribe" else all_special_ids[-6]
        )

    return Tokenizer(tokenizer=tokenizer, language=language, sot_sequence=tuple(sot_sequence))



