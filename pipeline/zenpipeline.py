
from zenml.integrations.constants import TENSORFLOW, MLFLOW
from zenml.pipelines import pipeline


@pipeline
def load_spectrogram_pipeline(
    get_words,
    spectrogram_producer
):
    """Links the get_words and spectrogram_producer steps together in a pipeline"""
    spectrogram_producer(get_words())


@pipeline(
    required_integrations=[TENSORFLOW],
    requirements_file="pipeline-requirements.txt",
    enable_cache=True,
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
    requirements_file="pipeline-requirements.txt",
    enable_cache=True,
)
def train_evaluate_and_deploy_pipeline(
    get_words,
    spectrogram_producer,
    lstm_trainer,
    keras_evaluator,
    deployment_trigger,
    model_deployer
):
    """Links the get_words and spectrogram_producer steps together in a pipeline"""
    X_train, X_test, y_train, y_test, timesteps = spectrogram_producer(get_words())
    model = lstm_trainer(X_train=X_train, y_train=y_train, timesteps=timesteps)
    loss, accuracy = keras_evaluator(X_test=X_test, y_test=y_test, model=model)
    deployment_decision = deployment_trigger(accuracy=accuracy)
    model_deployer(deployment_decision)


@pipeline(
    required_integrations=[TENSORFLOW, MLFLOW],
    requirements_file="pipeline-requirements.txt",
    enable_cache=True,
)
def dvc_train_evaluate_and_deploy_pipeline(
    get_paths_by_file,
    dvc_load_spectrograms,
    preprocess_spectrograms,
    lstm_trainer,
    keras_evaluator,
    deployment_trigger,
    model_deployer
):
    """Links the get_words and spectrogram_producer steps together in a pipeline"""
    hello_words, goodbye_words = get_paths_by_file()
    X, y, timesteps = dvc_load_spectrograms(hello_words=hello_words, goodbye_words=goodbye_words)
    X_train, X_test, y_train, y_test = preprocess_spectrograms(X=X, y=y, maximum_X=timesteps)
    model = lstm_trainer(X_train=X_train, y_train=y_train, timesteps=timesteps)
    loss, accuracy = keras_evaluator(X_test=X_test, y_test=y_test, model=model)
    deployment_decision = deployment_trigger(accuracy=accuracy)
    model_deployer(deployment_decision)


@pipeline(
    required_integrations=[TENSORFLOW, MLFLOW],
    requirements_file="pipeline-requirements.txt",
    enable_cache=True,
)
def dvc_train_evaluate_pipeline(
    get_paths_by_file,
    dvc_load_spectrograms,
    preprocess_spectrograms,
    lstm_trainer,
    keras_evaluator
):
    """Links the get_words and spectrogram_producer steps together in a pipeline"""
    hello_words, goodbye_words = get_paths_by_file()
    X, y, timesteps = dvc_load_spectrograms(hello_words=hello_words, goodbye_words=goodbye_words)
    X_train, X_test, y_train, y_test = preprocess_spectrograms(X=X, y=y, maximum_X=timesteps)
    model = lstm_trainer(X_train=X_train, y_train=y_train, timesteps=timesteps)
    keras_evaluator(X_test=X_test, y_test=y_test, model=model)
