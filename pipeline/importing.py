import os
from zenml.steps import Output, step
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

@step
def get_words() -> Output(
    words=np.ndarray
):
    """Returns paths to folders with example data for each word"""
    return np.array(["audio/hello", "audio/goodbye"])


def spectrogram_from_file(file_path, max_freq=10000):
    samples, sampling_rate = librosa.load(file_path, sr=None, mono=True, offset=0.0, duration=None)
    X = librosa.stft(samples)
    return librosa.amplitude_to_db(abs(X))


def spectrograms_from_folder(folder_path, tag):
    X = [spectrogram_from_file(f"{folder_path}/{name}") for name in os.listdir(folder_path)]
    y = [tag] * len(X)
    return X, y


@step
def load_spectrograms_from_audio(
        words: np.ndarray
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

    maximum_X = max([i.shape[1] for i in X])
    # Pad the values of X with 0s upto the maximum time steps and transpose the matrix
    X = [np.pad(i, [(0, 0), (0, maximum_X - i.shape[1])], constant_values=(0,)).T for i in X]

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
