from zenml.integrations.constants import TENSORFLOW, MLFLOW
from zenml.pipelines import pipeline
from zenml.environment import Environment
from zenml.integrations.mlflow.mlflow_environment import MLFLOW_ENVIRONMENT_NAME

from evaluating import keras_evaluator
from importing import get_words, load_spectrograms_from_audio, load_spectrogram_from_file
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

@pipeline(
    required_integrations=[TENSORFLOW, MLFLOW],
    requirements_file="pipeline-requirements.txt"
)
def mlflow_pipeline(
    get_words,
    spectrogram_producer,
    lstm_trainer,
    keras_evaluator
):
    """Links the get_words and spectrogram_producer steps together in a pipeline"""
    X_train, X_test, y_train, y_test, timesteps = spectrogram_producer(get_words())
    model = lstm_trainer(X_train=X_train, y_train=y_train, timesteps=timesteps)
    keras_evaluator(X_test=X_test, y_test=y_test, model=model)


pipeline1 = mlflow_pipeline(
    get_words=get_words(),
    spectrogram_producer=load_spectrograms_from_audio(),
    lstm_trainer=lstm_trainer(config=LSTMConfig(epochs=10)),
    keras_evaluator=keras_evaluator()
)

pipeline1.run()

pipeline2 = mlflow_pipeline(
    get_words=get_words(),
    spectrogram_producer=load_spectrograms_from_audio(),
    lstm_trainer=lstm_trainer(config=LSTMConfig(epochs=20)),
    keras_evaluator=keras_evaluator()
)

pipeline2.run()

mlflow_env = Environment()[MLFLOW_ENVIRONMENT_NAME]
print(
    "Now run \n "
    f"    mlflow ui --backend-store-uri {mlflow_env.tracking_uri}\n"
    "To inspect your experiment runs within the mlflow ui.\n"
    "You can find your runs tracked within the `mlflow_example_pipeline`"
    "experiment. Here you'll also be able to compare the two runs.)"
)
