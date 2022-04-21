
from google.protobuf.duration_pb2 import Duration
from feast import Entity, Feature, FeatureView, FileSource, ValueType

audio_files = FileSource(
    path="/home/ollie/FuzzyLabs/mlops/voice-recognition/parquet/audio_files.parquet",
    event_timestamp_column="event_timestamp",
)

audio = Entity(name="audio_id", value_type=ValueType.INT64, description="Audio ID",)

audio_files_view = FeatureView(
    name="audio_files",
    entities=["audio_id"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="audio_id", dtype=ValueType.INT64),
        Feature(name="file_path", dtype=ValueType.STRING),
        Feature(name="spectrogram_bytes", dtype=ValueType.BYTES),
        Feature(name="spectrogram_extended_bytes", dtype=ValueType.BYTES),
        Feature(name="audio_bytes", dtype=ValueType.BYTES),
        Feature(name="transcript", dtype=ValueType.STRING),
    ],
    online=True,
    batch_source=audio_files,
    tags={},
)
