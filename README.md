# Voice Recognition

## Project Motivation
I love the idea of virtual assistants, as a sci-fi fan and a gadget lover Amazon's [Alexa]() and Google's [Home]() are
another step into the Tony Stark / Star Trek gadget utopia that can help us work, learn and play better.

But I do have two issues, firstly, what's out there for privacy freaks? If I'm keeping something like that in my home I 
don't just want a guarantee that it's not being used for anything nefarious, I don't just want it disconnected from the
internet, I want to be able to see, tweak and learn from every line of code that goes into its creation.

Secondly, security, to achieve all the functionality I want for my assistant I want to be able to pay for things,
access my accounts on streaming services like Spotify or Netflix and even lock and unlock my front door. I want to 
fundamentally bake into the system that only the people I choose that can do these things.

####So how do we achieve this?
This project is a completely open source mlops pipeline that lets you create a voice recognition model, based on training
data of your own voice. For example to get the model to recognise `hello` and `goodbye` you record yourself saying those
words ~100 times and label them, then feed them into the training of our model.

The project will produce a model that will label similar audio files and can be deployed on the edge.
TODO: This should be updated over time as the project is fleshed out

## Blog Ideas / Stages
- Data Exploration / Initial model creation
- Experiment tracking and model improvement
- Model deployment

# Usage
Enter the virtual environment with `pipenv shell`
Install dependencies with `pipenv install`

Download data with `dvc pull`, if you want to push to dvc you need to authenticate:
```shell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <DagsHub-user-name>
dvc remote modify origin --local password <Token>
```

To swap to a different branch, first switch to that branch in git, e.g.:
`git checkout master`
Then checkout the data with dvc:
`dvc checkout`
