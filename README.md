## Overview

This is a project I worked on as part of my Machine Leearning course. The original project is available
[here](https://github.com/jeremyjordan/machine-learning/tree/master/projects/dog-project).

## Data

The training data for this project is located [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
This dataset contains 133 different breeds of dogs and is already split into train, test, and validation sets.
Place the training, testing, and validation datasets in the `dogImages` folder.

## Instructions

If you want to run the neural network, follow these instructions:
* download [DogInceptionV3Data.npz](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz)
which consists of pretrained InceptionV3 model. This will significantly reduce training time
* look at `requirements.txt`
* run `python dog_app.py -h` to see more information about training and predicting dog's breed using this NN
