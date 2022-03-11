# Voice Recognition

## Project Motivation
I love the idea of virtual assistants, as a sci-fi fan and a gadget lover Amazon's [Alexa]() and Google's [Home]() are
another step into the Tony Stark / Star Trek gadget utopia that can help us work, learn and play better.

But I do have two issues, firstly, what's out there for privacy freaks? If I'm keeping something like that in my home I 
don't just want a guarantee that it's not being used for anything nefarious, I don't just want it disconnected from the
internet, I want to be able to see, tweak and learn from every line of code that goes into its creation.

Secondly, I want to be able to design the device so that fundamentally only my voice, or the voice of those I choose,
can trigger commands. This is achieved by specifying training data that is unique to my voice and has a number of
benefits. I can now pay for things, access my accounts on streaming services like Spotify or Netflix and even lock and
unlock my front door, knowing that my device can only be operated by me.

#### So how do we achieve this?
This project is a completely open source mlops pipeline that lets you create a voice recognition model, based on training
data of your own voice. For example to get the model to recognise `hello` and `goodbye` you record yourself saying those
words ~30 times and label them, then feed them into the training of our model.

The project will produce a model that will label similar audio files and can be deployed on the edge.

This project uses the following open source tools:
* [ZenML](https://zenml.io/) - `pipeline/` is a ZenML pipeline, see [usage](#usage) for instructions to run it
* [DVC](https://dvc.org/) - The audio dataset is stored with DVC under a [DagsHub](https://dagshub.com/dashboard) remote
[here](https://dagshub.com/fuzzylabs/voice-recognition), this syncs with the GitHub repository. See [usage](#usage) for
instructions on acquiring and contributing to the dataset.


# Usage
Set up the virtual environment with
```shell
cd pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
zenml integration install tensorflow
```

Download the audio dataset with `dvc pull -r origin`.

N.b. To get the latest version of the dataset `dvc pull -r origin` should be run whenever you change branch.

To run the ZenML pipeline the following in the `pipeline` directory:
```shell
python zenpipeline.py
```

Now running the following will return a summary and an accuracy for the last pipeline that was run
```shell
python inspector.py
```

## Making changes to the dataset
To make changes to the dataset: first, authenticate:
```shell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <DagsHub-user-name>
dvc remote modify origin --local password <Token>
```

Then add, commit and push your changes like so:
```shell
dvc add <file-path>
git add .
git commit -m"Updated the dataset"
dvc push -r origin
git push
```

## Kubeflow
The Kubeflow integration requires [K3D](https://k3d.io/v5.2.1/#installation) and [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)
to be installed before it will function correctly.

To install the Kubeflow integration run:
```shell
zenml integration install kubeflow
```

To run the pipeline with Kubeflow enabled:
```shell
docker build . -t my_custom_zen_image
zenml stack set local_kubeflow_stack
zenml stack up
```
This will display a link to access your local kubeflow dashboard, if you perform a run with `python zenpipeline.py`
you can see the pipeline in kubeflow.

Once you're done run `zenml stack down` to clear down the kubernetes clusters.
It has happened before that the cluster is not cleared down correctly, if that is the case you can use k3d to clear the cluster(s)
```shell
k3d cluster list
k3d cluster delete <name>
```