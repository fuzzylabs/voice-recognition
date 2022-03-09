import numpy as np
import pandas as pd
from zenml.repository import Repository
from zenml.steps import Output, step
from zenml.integrations.facets.visualizers.facet_statistics_visualizer import (
    FacetStatisticsVisualizer,
)


def visualize_statistics():
    repo = Repository()
    pipe = repo.get_pipelines()[-1]
    importer_outputs = pipe.runs[-1].get_step(name="visualize_epochs")
    FacetStatisticsVisualizer().visualize(importer_outputs)


@step
def visualize_epochs(
    losses: np.ndarray, accuracies: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        [[losses[i], accuracies[i]] for i, _ in enumerate(losses)],
        columns=["Losses", "Accuracies"]
    )


if __name__ == "__main__":
    visualize_statistics()
