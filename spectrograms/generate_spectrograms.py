# Helper script which generates spectrograms from /audio and saves them to /spectrograms
import os

import librosa
import numpy as np


def spectrogram_from_file(file_path, max_freq=10000):
    samples, sampling_rate = librosa.load(file_path, sr=None, mono=True, offset=0.0, duration=None)
    X = librosa.stft(samples)
    return librosa.amplitude_to_db(abs(X))


def spectrograms_from_folder(folder_path, tag):
    X = [spectrogram_from_file(f"{folder_path}/{name}") for name in os.listdir(folder_path)]
    y = [tag] * len(X)
    return X, y


def get_spectrograms_from_audio():
    X = []
    y = []
    for i, folder_path in enumerate([f"../audio/{directory}" for directory in os.listdir("../audio") if os.path.isdir(f"../audio/{directory}")]):
        folder_X, folder_y = spectrograms_from_folder(folder_path, i)
        X += folder_X
        y += folder_y

    maximum_X = max([i.shape[1] for i in X])
    # Pad the values of X with 0s upto the maximum time steps and transpose the matrix
    X = [np.pad(i, [(0, 0), (0, maximum_X - i.shape[1])], constant_values=(0,)).T for i in X]

    return np.array(X), np.array(y)


# [spectrograms_from_folder(f"../audio/{name}", name) for name in os.listdir("../audio") if os.path.isdir(f"../audio/{name}")]

spectrograms, labels = get_spectrograms_from_audio()

np.save("spectrograms", spectrograms)
np.save("labels", labels)

