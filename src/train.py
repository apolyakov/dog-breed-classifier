import numpy as np
from tensorflow import keras

from . import utils


def main():
    print('Loading dataset...')
    (train_InceptionV3, valid_InceptionV3, test_InceptionV3),\
    (train_files, train_targets), (valid_files, valid_targets),\
    (test_files, test_targets) = utils.get_dataset()

    checkpointer = keras.callbacks.ModelCheckpoint(filepath=utils.SAVED_MODEL,
                                                   verbose=1, save_best_only=True)

    model = utils.create_model(train_InceptionV3.shape[1:])

    # train the model.
    model.fit(train_InceptionV3, train_targets,
              validation_data=(valid_InceptionV3, valid_targets),
              epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

    # get index of predicted dog breed for each image in test set.
    test_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]

    # report test accuracy.
    test_accuracy = 100 * np.sum(np.array(test_predictions) == np.argmax(test_targets, axis=1)) / len(test_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    return model
