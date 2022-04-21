from datetime import datetime
import os
from typing import Dict
import pandas as pd
import librosa
import numpy as np


def spectrogram_from_samples(samples):
    X = librosa.stft(samples)
    return librosa.amplitude_to_db(abs(X))


def spectrogram_from_file(file_path):
    samples, _ = librosa.load(file_path, sr=None, mono=True, offset=0.0, duration=None)
    return spectrogram_from_samples(samples)


def prep_spectrogram(spectrogram: np.array, new_timesteps: int = None) -> np.array:
    # Pad the values of X with 0s up to the given time steps
    if new_timesteps is None:
        return spectrogram
    return np.pad(spectrogram, [(0, 0), (0, new_timesteps - spectrogram.shape[1])], constant_values=(0,)).T


def save_dict_to_parquet(dict: Dict, save_path: str):
    pd.DataFrame.from_dict(
        dict
    ).to_parquet(save_path)


dictionary = {
    "audio_id": [],
    "audio_bytes": [],
    "transcript": [],
    "file_path": [],
    "spectrogram_bytes": [],
    "spectrogram_extended_bytes": [],
    "event_timestamp": [],
}


def add_audio(dictionary: Dict, audio_id: int, path: str, label: str):
    with open(path, "rb") as wav_file:
        dictionary["audio_id"].append(audio_id)
        dictionary["audio_bytes"].append(wav_file.read())
        dictionary["transcript"].append(label)
        dictionary["file_path"].append(path)
        dictionary["spectrogram_bytes"].append(spectrogram_from_file(path).tobytes())
        dictionary["spectrogram_extended_bytes"].append(
            prep_spectrogram(spectrogram_from_file(path), new_timesteps=200).tobytes()
        )
        dictionary["event_timestamp"].append(datetime.fromtimestamp(os.stat(path).st_mtime))


audio_id = 0

for path in [f"../audio/hello/{i}.wav" for i in range(1, 31)]:
    add_audio(dictionary, audio_id, path, "hello")
    audio_id += 1

for path in [f"../audio/goodbye/{i}.wav" for i in range(1, 31)]:
    add_audio(dictionary, audio_id, path, "goodbye")
    audio_id += 1

save_dict_to_parquet(dictionary, "audio_files.parquet")
