import click

from evaluating import keras_hparam_evaluator
from importing import get_words, load_spectrograms_from_audio
from training import lstm_hparam_trainer, LSTMConfig
from zenpipeline import train_and_evaluate_pipeline
from tensorboard.plugins.hparams import api as hp

from zenml.integrations.tensorflow.visualizers import (
    visualize_tensorboard,
    stop_tensorboard_server,
)


def get_pipeline(config: LSTMConfig):
    return train_and_evaluate_pipeline(
        get_words=get_words(),
        spectrogram_producer=load_spectrograms_from_audio(),
        lstm_trainer=lstm_hparam_trainer(config=config),
        keras_evaluator=keras_hparam_evaluator()
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

    HP_EPOCHS = hp.HParam('epochs', hp.IntInterval(10, 100))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.IntInterval(5, 10))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))

    session_num = 0

    for epochs in (HP_EPOCHS.domain.min_value, HP_EPOCHS.domain.max_value):
        for batch_size in (HP_BATCH_SIZE.domain.min_value, HP_BATCH_SIZE.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "optimizer": optimizer,
                })
                get_pipeline(LSTMConfig(
                    epochs=epochs,
                    batch_size=batch_size,
                    optimizer=optimizer,
                    loss="mean_squared_error",
                ))
                session_num += 1

    visualize_tensorboard(
        pipeline_name="train_and_evaluate_pipeline",
        step_name="lstm_trainer",
    )


if __name__ == "__main__":
    main()
