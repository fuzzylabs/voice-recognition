import io
import os

import soundfile
from zenml.steps import Output, step, BaseStepConfig
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

import dvc.api


@step
def get_words() -> Output(
    words=np.ndarray
):
    """Returns paths to folders with example data for each word"""
    return np.array(["audio/hello", "audio/goodbye"])

@step
def get_paths_by_file() -> Output(
    hello_words=np.ndarray,
    goodbye_words=np.ndarray
):
    """Returns paths to all of the wav files in each folder"""
    return np.array([f"audio/hello/{i}.wav" for i in range(1, 31)]), np.array([f"audio/goodbye/{i}.wav" for i in range(1, 31)])


def spectrogram_from_samples(samples):
    X = librosa.stft(samples)
    return librosa.amplitude_to_db(abs(X))


def spectrogram_from_file(file_path):
    samples, _ = librosa.load(file_path, sr=None, mono=True, offset=0.0, duration=None)
    return spectrogram_from_samples(samples)


def wav_bytes_to_spectrogram(input_bytes: bytes):
    samples, _ = librosa.load(soundfile.SoundFile(io.BytesIO(input_bytes)))
    return spectrogram_from_samples(samples)


def spectrograms_from_folder(folder_path, tag):
    X = [spectrogram_from_file(f"{folder_path}/{name}") for name in os.listdir(folder_path)]
    y = [tag] * len(X)
    return X, y


def prep_spectrogram(spectrogram: np.array, new_timesteps: int = None) -> np.array:
    # Pad the values of X with 0s up to the given time steps
    if new_timesteps is None:
        return spectrogram
    return np.pad(spectrogram, [(0, 0), (0, new_timesteps - spectrogram.shape[1])], constant_values=(0,)).T


def spectrogram_from_dvc(file_path):
    wav_bytes = dvc.api.read(
        path=file_path,
        repo='https://github.com/fuzzylabs/voice-recognition.git',
        remote="origin",
        mode="rb",
    )
    return wav_bytes_to_spectrogram(wav_bytes)


class LoadSpectrogramConfig(BaseStepConfig):
    # N.B. Timesteps must be more than the maximum X value of the desired inputs
    max_timesteps: int = None


@step
def load_spectrograms_from_audio(
    words: np.ndarray,
    config: LoadSpectrogramConfig
) -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray, timesteps=int
):
    """Loads the wav files as spectrograms and split them into training and testing split"""
    X = []
    y = []
    for i, folder_path in enumerate(words):
        folder_X, folder_y = spectrograms_from_folder(folder_path, i)
        X += folder_X
        y += folder_y

    if config.max_timesteps is None:
        maximum_X = max([i.shape[1] for i in X])
    else:
        maximum_X = config.max_timesteps

    # Pad the values of X with 0s upto the maximum time steps and transpose the matrix
    X = [prep_spectrogram(i, maximum_X) for i in X]

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, maximum_X


@step
def load_spectrogram_from_file() -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray, timesteps=int
):
    """Loads the spectrograms saved in an np array directly from ../spectrograms/spectrograms.npy"""
    X, y = np.load("../spectrograms/spectrograms.npy"), np.load("../spectrograms/labels.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, max([i.shape[1] for i in X])


@step
def dvc_load_spectrograms(
    hello_words: np.ndarray,
    goodbye_words: np.ndarray,
    config: LoadSpectrogramConfig,
) -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray, timesteps=int
):
    """Loads the wav files from DVC as spectrograms and split them into training and testing split"""
    X = []
    y = []
    for i, file_path in enumerate(hello_words):
        X += [spectrogram_from_dvc(file_path)]
        y += [0]
    for i, file_path in enumerate(goodbye_words):
        X += [spectrogram_from_dvc(file_path)]
        y += [1]

    if config.max_timesteps is None:
        maximum_X = max([i.shape[1] for i in X])
    else:
        maximum_X = config.max_timesteps

    # Pad the values of X with 0s upto the maximum time steps and transpose the matrix
    X = [prep_spectrogram(i, maximum_X) for i in X]

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, maximum_X