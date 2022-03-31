import numpy as np
import librosa
import soundfile
import io
import mlflow
import keras
import base64


class AudioClassifier(mlflow.pyfunc.PythonModel):

    def spectrogram_from_samples(self, samples):
        X = librosa.stft(samples)
        return librosa.amplitude_to_db(abs(X))

    def prep_spectrogram(self, spectrogram: np.array, new_timesteps: int = None) -> np.array:
        # Pad the values of X with 0s up to the given time steps
        if new_timesteps is None:
            return spectrogram
        return np.pad(spectrogram, [(0, 0), (0, new_timesteps - spectrogram.shape[1])], constant_values=(0,)).T

    def wav_bytes_to_spectrogram(self, input_bytes: bytes, timesteps=None):
        samples, _ = librosa.load(soundfile.SoundFile(io.BytesIO(input_bytes)))
        return self.prep_spectrogram(self.spectrogram_from_samples(samples), timesteps)

    def load_context(self, context):
        self.model = keras.models.load_model(context.artifacts["model"])

    # TODO: Make this function a parameter of the classifier
    def pre_process_input(self, model_input):
        return np.array([
            self.wav_bytes_to_spectrogram(base64.b64decode(model), 200) for model in model_input
        ])

    def predict(self, context, model_input):
        # Predict takes a list of base64 encoded .wav files
        # converts them to spectrograms and returns the inferred prediction for each

        return self.model.predict(self.pre_process_input(model_input))
