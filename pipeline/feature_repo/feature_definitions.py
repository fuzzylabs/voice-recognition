# This is an example feature definition file

from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, FileSource, ValueType

audio_files = FileSource(
    path="/home/ollie/FuzzyLabs/mlops/voice-recognition/parquet/audio_files.parquet",
    event_timestamp_column="event_timestamp",
)

driver = Entity(name="audio_id", value_type=ValueType.INT64, description="Audio ID",)

audio_files_view = FeatureView(
    name="audio_files",
    entities=["audio_id"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="raw_audio", dtype=ValueType.BYTES),
        Feature(name="label", dtype=ValueType.STRING),
    ],
    online=True,
    batch_source=audio_files,
    tags={},
)
