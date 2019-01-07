## Overview

This is a project I worked on as part of my Machine Leearning course. The original project is available
[here](https://github.com/jeremyjordan/machine-learning/tree/master/projects/dog-project).  
The average `test_accuracy` for this network is about 80%.

## Data

The training data for this project is located [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
This dataset contains 133 different breeds of dogs and is already split into train, test, and validation sets.

## Instructions

To run the neural network, follow these instructions:
* download the dataset and place it under `path/to/project` folder
* download [DogInceptionV3Data.npz](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz)
which consists of pretrained InceptionV3 model and place it under `path/to/project/bottleneck_features/` folder. This will significantly reduce training time
* obtain the necessary Python packages (see `requirements.txt`) and switch Keras backend to Tensorflow
  ```
  KERAS_BACKEND=tensorflow python -c "from keras import backend"
  ```
* run `python dog_app.py -h` to see more information about training and predicting dog's breed using this NN
