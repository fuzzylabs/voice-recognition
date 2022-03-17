# Open Source Voice Assistants
### closed-circuit, super-visible, completely personalised

![Scotty from Star Trek tries to use a mouse as if you can speak into it](Scotty_uses_a_mouse.jpg)

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

The project will produce a model that will classify similar audio files and can be deployed on the edge.

This project uses the following open source tools:
* [ZenML](https://zenml.io/) - `pipeline/` is a ZenML pipeline, see [usage](#usage) for instructions to run it
* [DVC](https://dvc.org/) - The audio dataset is stored with DVC under a [DagsHub](https://dagshub.com/dashboard) remote
[here](https://dagshub.com/fuzzylabs/voice-recognition), this syncs with the GitHub repository. See [usage](#usage) for
instructions on acquiring and contributing to the dataset.


# Usage
### Setup virtual environment
```shell
cd pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Setup ZenML
```shell
zenml init
zenml integration install tensorflow
```

### Setup DVC
Download the audio dataset with `dvc pull -r origin`.

N.b. To get the latest version of the dataset `dvc pull -r origin` should be run whenever you change branch.

To run the ZenML pipeline run the following in the `pipeline` directory:
```shell
python zenpipeline.py
```
This runs the pipeline, which trains the model and stores the artifacts from the pipeline in the local store.


Now running the following will return a summary and an accuracy for the last pipeline that was run
```shell
python inspector.py
```

#### Making changes to the dataset
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

## Swapping to a different branch
To swap to a different branch, first switch to that branch in git, e.g.:
`git checkout master`
Then checkout the data with dvc:
`dvc checkout`

## Setup Kubeflow

Note: ZenML's Kubeflow integration is still experimental and as a result the steps to make this work are a bit cumbersome
but as the integration irons out bugs this workflow will improve

The Kubeflow integration requires [K3D](https://k3d.io/v5.2.1/#installation) and [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)
to be installed before it will function correctly.

To install the Kubeflow integration, run:
```shell
zenml integration install kubeflow
```

Then set up the local container registry and local kubeflow orchestrator and combine them into the Kubeflow stack:
```shell
# Make sure to create the local registry on port 5000 for it to work 
zenml container-registry register local_registry --type=default --uri=localhost:5000 
zenml orchestrator register kubeflow_orchestrator --type=kubeflow
zenml stack register local_kubeflow_stack \
    -m local_metadata_store \
    -a local_artifact_store \
    -o kubeflow_orchestrator \
    -c local_registry

# Activate the newly created stack
zenml stack set local_kubeflow_stack
```

Then open `/.zen/orchestrators/kubeflow_orchestrator.yaml` and update `custom_docker_base_image_name` to `base_zenml_image`

To run the pipeline with Kubeflow enabled:
```shell
docker build . -t base_zenml_image
zenml stack up
```
This will display a link to access your local kubeflow dashboard, if you perform a run with `python zenpipeline.py`
you can see the pipeline in kubeflow.

Set `custom_docker_base_image_name` in .zen/orchestrators/kubeflow_orchestrator.yaml to `base_zenml_image`

Once you're done run `zenml stack down` to clear down the kubernetes clusters.
It has happened before that the cluster is not cleared down correctly, if that is the case you can use k3d to clear the cluster(s)
```shell
k3d cluster list
k3d cluster delete <name>
```