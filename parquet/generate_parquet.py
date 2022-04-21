from datetime import datetime
import os
from typing import Dict
import pandas as pd


def save_dict_to_parquet(dict: Dict, save_path: str):
    pd.DataFrame.from_dict(
        dict
    ).to_parquet(save_path)


dictionary = {
    "audio_id": [],
    "raw_audio": [],
    "label": [],
    "event_timestamp": [],
}


def add_audio(dictionary: Dict, audio_id: int, path: str, label: str):
    with open(path, "rb") as wav_file:
        dictionary["audio_id"].append(audio_id)
        dictionary["raw_audio"].append(wav_file.read())
        dictionary["label"].append(label)
        dictionary["event_timestamp"].append(datetime.fromtimestamp(os.stat(path).st_mtime))


audio_id = 0

for path in [f"../audio/hello/{i}.wav" for i in range(1, 31)]:
    add_audio(dictionary, audio_id, path, "hello")
    audio_id += 1

for path in [f"../audio/goodbye/{i}.wav" for i in range(1, 31)]:
    add_audio(dictionary, audio_id, path, "goodbye")
    audio_id += 1

save_dict_to_parquet(dictionary, "audio_files.parquet")
