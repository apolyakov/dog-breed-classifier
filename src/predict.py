import os

import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input

import train, utils


def main(image_path):
    print('Loading the image...')
    image_tensor = utils.img_to_tensor(image_path)

    print('Extracting InceptionV3 features for the image...')
    bottleneck_features = InceptionV3(weights='imagenet', include_top=False).predict(
        preprocess_input(image_tensor))

    # Preparing the model.
    if os.path.isfile(utils.SAVED_MODEL):
        # Load the trained model if we have it.
        model = utils.load_model(utils.SAVED_MODEL)
    else:
        # Train the model otherwise.
        model = train.main()

    print('\nPredicting...')
    prediction = model.predict(bottleneck_features)[0]

    top_N = 4
    dog_names = utils.load_dog_names()

    # sort predicted breeds by highest probability, extract the top N predictions
    breeds_predicted = [dog_names[idx].replace('_', ' ') for idx in np.argsort(prediction)[::-1][:top_N]]
    confidence_predicted = np.sort(prediction)[::-1][:top_N]

    print('It looks like a %s.' % breeds_predicted[0])
    print('\nTop 4 predictions:')
    for breed, confidence in zip(breeds_predicted, confidence_predicted):
        print('Predicted breed: %s with a confidence of %.2f%%' % (breed, confidence * 100))
