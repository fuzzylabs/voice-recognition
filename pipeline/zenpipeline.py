from zenml.integrations.constants import TENSORFLOW
from zenml.pipelines import pipeline

from evaluating import keras_evaluator
# from importing import get_words, load_spectrograms_from_audio
from importing import load_spectrogram_from_file
from training import lstm_trainer, LSTMConfig


@pipeline
def load_spectrogram_pipeline(
    get_words,
    spectrogram_producer
):
    """Links the get_words and spectrogram_producer steps together in a pipeline"""
    spectrogram_producer(get_words())


@pipeline(
    required_integrations=[TENSORFLOW],
    requirements_file="pipeline-requirements.txt"
)
def train_and_evaluate_pipeline(
    get_words,
    spectrogram_producer,
    lstm_trainer,
    keras_evaluator
):
    """Links the get_words and spectrogram_producer steps together in a pipeline"""
    X_train, X_test, y_train, y_test, timesteps = spectrogram_producer(get_words())
    model = lstm_trainer(X_train=X_train, y_train=y_train, timesteps=timesteps)
    keras_evaluator(X_test=X_test, y_test=y_test, model=model)

@pipeline(
    required_integrations=[TENSORFLOW],
    requirements_file="pipeline-requirements.txt"
)
def train_and_evaluate_preloaded_spectrogram_pipeline(
    spectrogram_producer,
    lstm_trainer,
    keras_evaluator
):
    """Links the get_words and spectrogram_producer steps together in a pipeline"""
    X_train, X_test, y_train, y_test, timesteps = spectrogram_producer()
    model = lstm_trainer(X_train=X_train, y_train=y_train, timesteps=timesteps)
    keras_evaluator(X_test=X_test, y_test=y_test, model=model)


pipeline = train_and_evaluate_preloaded_spectrogram_pipeline(
    spectrogram_producer=load_spectrogram_from_file(),
    lstm_trainer=lstm_trainer(config=LSTMConfig()),
    keras_evaluator=keras_evaluator()
)

pipeline.run()
