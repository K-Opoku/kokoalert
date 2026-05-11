import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from src.config import (
    SAMPLE_RATE, WINDOW_SECONDS, HOP_SECONDS,
    N_MELS, N_FFT, HOP_LENGTH, FMIN, FMAX, SPEC_TIME_STEPS
)


# ── STEP 1: LOAD AND NORMALISE ────────────────────────────────────────────────

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to our target sample rate.

    librosa.load() does two things at once:
    1. Decodes the audio (WAV, MP3, OGG all work)
    2. Resamples to the target sample rate

    We set mono=True because we only care about sound content,
    not stereo positioning. Two channels would double our data
    with no useful extra information for disease detection.

    Returns:
        y: Audio time series as numpy array
        sr: Sample rate (will always equal SAMPLE_RATE after resampling)
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def peak_normalise(y: np.ndarray) -> np.ndarray:
    """
    Scale audio so the loudest point is exactly 1.0 (or -1.0).

    Why this matters: A farmer recording from 2 metres away produces
    much quieter audio than one recording from 30cm. Without normalisation,
    the model learns loudness instead of sound patterns.

    The 1e-8 prevents division by zero on silent recordings.
    """
    peak = np.max(np.abs(y))
    return y / (peak + 1e-8)


# ── STEP 2: CONVERT TO MEL SPECTROGRAM ───────────────────────────────────────

def audio_to_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Convert a raw audio array to a log-mel spectrogram.

    The pipeline:
    1. librosa.feature.melspectrogram() runs a Short-Time Fourier Transform
       (STFT) and applies mel filterbank — gives us energy per mel band per
       time frame
    2. librosa.power_to_db() converts energy to decibels (log scale)
       This is critical — without it, a bird call at 0.001 energy and one
       at 1.0 energy look 1000x different in raw power but only 30dB
       different in log scale. The log scale matches how hearing works.

    Parameters from config.py:
    - N_MELS=128: Number of frequency bands (rows in the output image)
    - N_FFT=2048: FFT window size — higher = better frequency resolution
    - HOP_LENGTH=512: Samples between FFT windows — controls time resolution
    - FMIN=20, FMAX=8000: Frequency range — 20Hz is the floor of hearing,
      8000Hz captures all relevant chicken respiratory sounds

    Returns:
        Mel spectrogram shape: (128, time_frames) — a 2D array, like a
        greyscale image with 128 rows (frequencies) and N columns (time)
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


# ── STEP 3: WINDOW SLICING ────────────────────────────────────────────────────

def slice_into_windows(y: np.ndarray) -> list[np.ndarray]:
    """
    Slice a full audio clip into overlapping fixed-length windows.

    Why overlapping? Consider a 30-second clip with a coughing sound at
    second 12. A non-overlapping window might split that cough across two
    windows, weakening the signal in both. Overlapping windows guarantee
    every sound appears fully in at least one window.

    Window size: 5 seconds = 5 * 16000 = 80,000 samples
    Hop size: 2 seconds = 2 * 16000 = 32,000 samples

    A 30-second clip produces:
    windows = floor((30 - 5) / 2) + 1 = 13 windows

    Windows shorter than 5 seconds at the end are discarded — they
    would produce inconsistent spectrogram shapes that break the model.
    """
    window_samples = WINDOW_SECONDS * SAMPLE_RATE   # 80,000
    hop_samples = HOP_SECONDS * SAMPLE_RATE         # 32,000

    windows = []
    start = 0

    while start + window_samples <= len(y):
        window = y[start : start + window_samples]
        windows.append(window)
        start += hop_samples

    return windows


def fix_spectrogram_width(spec: np.ndarray) -> np.ndarray:
    """
    Force every spectrogram to exactly SPEC_TIME_STEPS columns.
    Uses reflect padding instead of constant — mirrors real audio content
    at the edge rather than inserting artificial silence, which would
    create the same border artifact we fixed in the CNN.
    In practice this rarely fires: a 5s window at 16kHz with center=True
    produces exactly 157 frames. This is a safety net only.
    """
    if spec.shape[1] < SPEC_TIME_STEPS:
        pad_width = SPEC_TIME_STEPS - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='reflect')
    elif spec.shape[1] > SPEC_TIME_STEPS:
        spec = spec[:, :SPEC_TIME_STEPS]
    return spec


# ── STEP 4: QUALITY CHECK ─────────────────────────────────────────────────────

def check_recording_quality(file_path: str) -> dict:
    """
    Check whether a recording is usable before analysis.

    Three failure modes we guard against:
    1. Too short — under 3 seconds means we can't even make one window
    2. Too quiet — RMS energy below threshold means the phone was too far
       away or the recording is essentially silence (mic covered, etc.)
    3. Clipping — if the waveform hits ±1.0 constantly, the recording is
       distorted and frequencies are unreliable

    RMS (Root Mean Square) energy is the standard measure of loudness.
    It averages the squared amplitude across all samples — a single loud
    spike doesn't dominate the way a peak measure would.

    These thresholds come from practical experience with field recordings:
    - min_duration: 3s is the minimum for one 5-second window to exist
      (we trim to 5s, so actually we need 5s — but 3s lets us give a
      more specific error message)
    - min_rms: 0.001 filters out recordings made with a covered microphone
      or recorder left running in an empty room
    - max_clip_ratio: 0.01 allows occasional natural peaks but rejects
      recordings where the signal is constantly saturated
    """
    try:
        y, sr = load_audio(file_path)
        duration = len(y) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(y ** 2)))
        clip_ratio = float(np.mean(np.abs(y) > 0.99))

        if duration < 3.0:
            return {
                "usable": False,
                "reason": "too_short",
                "duration": duration,
                "rms": rms,
            }

        if rms < 0.001:
            return {
                "usable": False,
                "reason": "too_quiet",
                "duration": duration,
                "rms": rms,
            }

        if clip_ratio > 0.01:
            return {
                "usable": False,
                "reason": "clipping",
                "duration": duration,
                "rms": rms,
                "clip_ratio": clip_ratio,
            }

        return {
            "usable": True,
            "duration": duration,
            "rms": rms,
            "clip_ratio": clip_ratio,
        }

    except Exception as e:
        return {
            "usable": False,
            "reason": "load_error",
            "error": str(e),
        }


# ── STEP 5: FULL PIPELINE FUNCTION ───────────────────────────────────────────

def file_to_spectrograms(file_path: str) -> list[np.ndarray]:
    """
    Full preprocessing pipeline: audio file → list of mel spectrograms.

    This is the function called by pipeline.py. It chains every step:
    load → normalise → slice windows → convert each window to spectrogram
    → fix shape → add channel dimension

    The channel dimension (np.newaxis at the end) reshapes from
    (128, 313) to (128, 313, 1). This is required by TensorFlow's
    Conv2D layers which expect (height, width, channels). We have one
    channel — like a greyscale image vs an RGB image with 3 channels.

    Returns:
        List of spectrograms, each shape (128, 313, 1)
        Empty list if file is too short or fails to load
    """
    try:
        y, _ = load_audio(file_path)
        y = peak_normalise(y)
        windows = slice_into_windows(y)

        spectrograms = []
        for window in windows:
            mel = audio_to_mel_spectrogram(window)
            mel = fix_spectrogram_width(mel)
            mel = mel[:, :, np.newaxis]   # (128, 313) → (128, 313, 1)
            spectrograms.append(mel)

        return spectrograms

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


# ── STEP 6: BATCH PROCESSING FOR TRAINING ────────────────────────────────────

def process_dataset_split(
    folder: str,
    label: int,
    max_clips: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process all audio files in a folder into spectrograms for training.

    Used in the training notebook to build your X_train and y_train arrays.

    Args:
        folder: Path to folder containing WAV files
        label: 0 = healthy, 1 = sick, 2 = noise
        max_clips: Optional limit — useful for quick experiments

    Returns:
        X: array of shape (N, 128, 313, 1) — N total windows
        y: array of shape (N,) — label repeated for every window

    Note on labels: For an autoencoder trained only on healthy data,
    you only use label=0 during training. Labels 1 and 2 are used
    during threshold calibration to verify sick clips score high.
    """
    all_specs = []
    all_labels = []

    files = [
    f for f in Path(folder).iterdir()
    if f.suffix.lower() in (".wav", ".ogg", ".mp3", ".m4a", ".mp4", ".flac")
]
    if max_clips:
        files = files[:max_clips]

    for i, f in enumerate(files):
        specs = file_to_spectrograms(str(f))
        all_specs.extend(specs)
        all_labels.extend([label] * len(specs))

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(files)} clips "
                  f"({len(all_specs)} windows so far)")

    if not all_specs:
        return np.array([]), np.array([])

    return np.array(all_specs), np.array(all_labels)