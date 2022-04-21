from datetime import datetime, timedelta
import pandas as pd

from feast import FeatureStore

# The entity dataframe is the dataframe we want to enrich with feature values
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
        "audio_files:raw_audio",
        "audio_files:label",
    ],
).to_df()

print("----- Feature schema -----\n")
print(training_df.info())

print()
print("----- Example features -----\n")
print(training_df.head())
