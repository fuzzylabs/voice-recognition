
from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Define an entity for the driver. You can think of entity as a primary key used to
# fetch features.
driver = Entity(name="audio_id", value_type=ValueType.INT64, description="Audio id",)

# Here we define a Feature View that will allow us to serve our data to our model online.
audio_files = FeatureView(
    name="audio_files",
    entities=["driver_id"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="raw_audio_file", dtype=ValueType.BYTES),
    ],
    online=True,
    tags={},
)
