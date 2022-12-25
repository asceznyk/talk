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




