import io
import os
import urllib
import hashlib
import warnings

from tqdm import tqdm
from typing import List, Optional, Union

import torch

from .audio import load_audio, log_mel_spec, pad_or_trim
from .decoding import DecodingOptions, DecodingResult, decode, detect_language
from .model import Talk, ModelDimensions
#from .transcribe import transcribe

def load_model(ckpt_path):
    print(f"loading checkpoint from {ckpt_path}..")
    with open(ckpt_path, "rb") as f: ckpt = torch.load(f) 
    model = Talk(ModelDimensions(**ckpt['dims']))
    model.load_state_dict(ckpt["model_state_dict"])
    status = f"successfully loaded checkpoint {ckpt_path}!"
    print(status)
    return model, status



