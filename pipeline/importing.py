import io
import os
import subprocess
from typing import List, Any
import pickle
from datetime import datetime
from typing import List, Any

import soundfile
from feast import FeatureStore
from zenml.steps import Output, step, BaseStepConfig
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

import dvc.api

from keras.preprocessing.image import ImageDataGenerator


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, maximum_X


@step
def load_spectrogram_from_file() -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray, timesteps=int
):
    """Loads the spectrograms saved in an np array directly from ../spectrograms/spectrograms.npy"""
    X, y = np.load("../spectrograms/spectrograms.npy"), np.load("../spectrograms/labels.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, max([i.shape[1] for i in X])


def prep_class_spectrograms(spectrograms: List[Any], labels: List[int], maximum_X: int, number_of_cases: int = 200):
    prepared_spectrograms = np.array([prep_spectrogram(s, new_timesteps=maximum_X) for s in spectrograms])

    reshaped_prepared_spectrograms = prepared_spectrograms.reshape(
        (prepared_spectrograms.shape[0], prepared_spectrograms.shape[1], prepared_spectrograms.shape[2], 1)
    )

    data_generator = ImageDataGenerator(width_shift_range=0.3)
    data_generator.fit(reshaped_prepared_spectrograms)
    X_iterator = data_generator.flow(reshaped_prepared_spectrograms, y=labels, batch_size=number_of_cases)

    Xs, ys = X_iterator.next()
    return Xs.reshape((Xs.shape[0], Xs.shape[1], Xs.shape[2])), ys


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
        y += [[1, 0]]
    for i, file_path in enumerate(goodbye_words):
        X += [spectrogram_from_dvc(file_path)]
        y += [[0, 1]]

    if config.max_timesteps is None:
        maximum_X = max([i.shape[1] for i in X])
    else:
        maximum_X = config.max_timesteps

    X, y = prep_class_spectrograms(X, y, maximum_X=maximum_X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, maximum_X


@step
def dvc_cli_load_spectrograms(
    hello_words: np.ndarray,
    goodbye_words: np.ndarray,
    config: LoadSpectrogramConfig,
) -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray, timesteps=int
):
    """Loads the wav files from DVC as spectrograms and split them into training and testing split"""

    if subprocess.call("dvc pull -r origin", shell=True) != 0:
        raise RuntimeError("dvc pull -r origin returned a non-zero status code")

    if subprocess.call("dvc checkout", shell=True) != 0:
        raise RuntimeError("dvc checkout returned a non-zero status code")

    X = []
    y = []
    for i, file_path in enumerate(hello_words):
        X += [spectrogram_from_file(f"../{file_path}")]
        y += [[1, 0]]
    for i, file_path in enumerate(goodbye_words):
        X += [spectrogram_from_file(f"../{file_path}")]
        y += [[0, 1]]

    if config.max_timesteps is None:
        maximum_X = max([i.shape[1] for i in X])
    else:
        maximum_X = config.max_timesteps

    X, y = prep_class_spectrograms(X, y, maximum_X=maximum_X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, maximum_X


def transcript_to_label(transcript: str) -> List[int]:
    if transcript == "hello":
        return [1, 0]
    elif transcript == "goodbye":
        return [0, 1]
    else:
        raise ValueError(f"No label value for {transcript}, expected 'hello' or 'goodbye'")


@step
def feast_load_spectrograms(
    config: LoadSpectrogramConfig,
) -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray, timesteps=int
):
    """Loads the extended spectrogram bytes from FEAST, decodes them and splits them into training and testing"""
    entity_df = pd.DataFrame.from_dict(
        {
            "audio_id": [0, 1, 2],
            "event_timestamp": [
                datetime(2022, 3, 25, 16, 15, 4, 548965),
                datetime(2022, 3, 25, 16, 15, 4, 676967),
                datetime(2022, 3, 25, 16, 15, 4, 700967),
            ],
        }
    )

    store = FeatureStore(repo_path="feature_repo")

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "audio_files:spectrogram_extended_bytes",
            "audio_files:transcript",
        ],
    ).to_df()

    X = np.array([
        pickle.loads(spectrogram_bytes) for spectrogram_bytes in training_df["spectrogram_extended_bytes"]
    ])
    y = np.array([transcript_to_label(transcript) for transcript in training_df["transcript"]])

    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, config.max_timesteps
