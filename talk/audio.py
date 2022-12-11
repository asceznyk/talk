import os
import ffmpeg
import numpy as np

from functools import lru_cache
from typing import Union

import torch
import torch.nn.functional as F

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input

def pad_or_trim(array:torch.Tensor, length:int = N_SAMPLES, *, axis:int=-1):
    if array.shape[axis] > length:
        array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
    elif array.shape[axis] < length:
        pad_widths = [(0,0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

    return array

def load_audio(file:str, sr:int = SAMPLE_RATE):
    try: 
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0 

@lru_cache(maxsize=None)
def mel_filters(device):
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{N_MELS}"]).to(device)

def log_mel_spec(audio:Union[str, torch.Tensor, np.ndarray]):
    if not torch.is_tensor(audio):
        if isinstance(audio, str): audio = load_audio(audio)
        audio = torch.from_numpy(audio)
 
    stft = torch.stft(
        audio, 
        N_FFT, 
        HOP_LENGTH, 
        window=torch.hann_window(N_FFT).to(audio.device),
        return_complex=True
    )

    log_spec = torch.clamp(mel_filters(audio.device) @ (stft[:, :-1].abs() ** 2), min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0




