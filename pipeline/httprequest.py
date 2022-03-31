import base64

import requests
from importing import wav_path_to_bytes

response = requests.post(
    "http://127.0.0.1:5000/invocations",
    json={"instances": [str(base64.b64encode(wav_path_to_bytes("audio/hello.wav")), "utf-8")]},
)

print(response.json())
