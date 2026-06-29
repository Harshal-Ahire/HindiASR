import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

TARGET_SAMPLE_RATE = 16000
MIN_SNR_THRESHOLD = 0.6

def resample_audio(audio_path: str, target_sr: int = TARGET_SAMPLE_RATE):
    """Resample audio to target sample rate."""
    audio, sr = librosa.load(audio_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def trim_silence(audio: np.ndarray, sr: int, top_db: int = 20):
    """Remove leading and trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def compute_snr(audio: np.ndarray) -> float:
    """Estimate signal-to-noise ratio for quality filtering."""
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(np.abs(np.diff(audio)) ** 2)
    return signal_power / (noise_power + 1e-10)

def preprocess_clip(audio_path: str) -> np.ndarray | None:
    """Full preprocessing pipeline for a single audio clip."""
    audio, sr = resample_audio(audio_path)
    audio = trim_silence(audio, sr)
    snr = compute_snr(audio)
    if snr < MIN_SNR_THRESHOLD:
        return None  # Filter low quality recordings
    return audio