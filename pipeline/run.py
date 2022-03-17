import click

from evaluating import keras_evaluator
from importing import get_words, load_spectrograms_from_audio
from training import LSTMConfig, lstm_trainer
from zenpipeline import train_and_evaluate_pipeline

from zenml.integrations.tensorflow.visualizers import (
    visualize_tensorboard,
    stop_tensorboard_server,
)


@click.command()
@click.option(
    "--stop-tensorboard",
    is_flag=True,
    default=False,
    help="Stop the Tensorboard server",
)
def main(stop_tensorboard: bool):

    if stop_tensorboard:
        stop_tensorboard_server(
            pipeline_name="train_and_evaluate_pipeline",
            step_name="lstm_trainer",
        )
        return

    pipeline = train_and_evaluate_pipeline(
        get_words=get_words(),
        spectrogram_producer=load_spectrograms_from_audio(),
        lstm_trainer=lstm_trainer(config=LSTMConfig(
            epochs=60,
            batch_size=10,
            optimizer="adam",
            loss="mean_squared_error",
        )),
        keras_evaluator=keras_evaluator()
    )

    pipeline.run()

    visualize_tensorboard(
        pipeline_name="train_and_evaluate_pipeline",
        step_name="lstm_trainer",
    )


if __name__ == "__main__":
    main()
