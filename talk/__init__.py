import hashlib
import io
import os
import urllib
import warnings
from typing import List, Optional, Union

import torch
from tqdm import tqdm

from .audio import load_audio, log_mel_spectrogram, pad_or_trim
#from .decoding import DecodingOptions, DecodingResult, decode, detect_language
from .model import Whisper, ModelDimensions
#from .transcribe import transcribe




