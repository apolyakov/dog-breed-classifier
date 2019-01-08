## Overview

This is a project I worked on as part of my Machine Leearning course. The original project is available
[here](https://github.com/jeremyjordan/machine-learning/tree/master/projects/dog-project).  
The average `test_accuracy` for this network is about 80%.

## Data

The training data for this project is located [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
This dataset contains 133 different breeds of dogs and is already split into train, test, and validation sets.

## Instructions

To run the neural network, follow these instructions:
* Download the features which consist of pretrained InceptionV3 model and place them under `path/to/project/bottleneck_features/` folder.
This will significantly reduce a training time. There are two ways of downloading the features:
  * manual way: download the features from [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz)
  * automatic way: the program will do an automatic download in case of missing features
* Obtain the necessary Python packages (see `requirements.txt`) and switch Keras backend to Tensorflow
  ```
  KERAS_BACKEND=tensorflow python -c "from keras import backend"
  ```
* Run `python dog_app.py -h` to see more information about training and predicting dog's breed using this NN

If you want to train the model yourself, you need to download the dataset and unpack it like `path/to/project/dogImages`.
Then follow the above described instructions.
