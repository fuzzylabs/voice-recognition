import base64

import requests
from importing import spectrogram_from_file, prep_spectrogram

response = requests.post(
    "http://localhost:8000/invocations",
    json={"instances": [
        prep_spectrogram(spectrogram_from_file("audio/hello.wav"), 250).tolist(),
    ]},
)

# print(["Hello" if float(i[0]) < 0.5 else "Goodbye" for i in response.json()])
print(response.json())
print(["Hello" if float(i[0]) < 0.5 else "Goodbye" for i in response.json()])
