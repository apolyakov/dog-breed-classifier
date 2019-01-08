import os

import numpy as np
from keras_preprocessing import image
from tensorflow import keras
from sklearn.datasets import load_files
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

THIS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
BOTTLENECK_FEATURES = os.path.join(THIS_DIRECTORY, '..', 'bottleneck_features', 'DogInceptionV3Data.npz')
BOTTLENECK_URL = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz'
DOG_IMAGES = os.path.join(THIS_DIRECTORY, '..', 'dogImages')
DOG_BREEDS = os.path.join(THIS_DIRECTORY, '..', 'resources', 'dog_breeds.txt')
DOG_BREEDS_COUNT = 133
IMAGE_SHAPE = (224, 224)
SAVED_MODEL = os.path.join(THIS_DIRECTORY, '..', 'saved_models', 'weights.best.InceptionV3.hdf5')


def img_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type.
    img = image.load_img(img_path, target_size=IMAGE_SHAPE)
    # convert PIL.Image.Image type to 3D tensor with shape (img_height, img_width, 3).
    img = image.img_to_array(img).astype(np.uint8)
    # convert 3D tensor to 4D tensor with shape (1, img_height, img_width, 3) and return 4D tensor.
    return np.expand_dims(img, axis=0).astype('float32')


def images_to_tensor(img_paths):
    list_of_tensors = [img_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# a function for loading train, test, and validation datasets.
def load_dataset(path):
    data = None
    try:
        data = load_files(path)
    except FileNotFoundError:
        print('Please, download the dogImages dataset.')
        exit()
    dog_files = np.array(data['filenames'])
    dog_targets = keras.utils.to_categorical(np.array(data['target']), DOG_BREEDS_COUNT)
    return dog_files, dog_targets


def download_bottleneck_url():
    import math
    import requests
    r = requests.get(BOTTLENECK_URL, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    with open(BOTTLENECK_FEATURES, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size / block_size), unit='KB', unit_scale=True):
            wrote += len(data)
            f.write(data)

    if total_size != 0 and wrote != total_size:
        print('Whoops, download went wrong.')
        exit()


def load_bottleneck_features(model_is_trained=False):
    # load train, test and validation features for InceptionV3.
    if not model_is_trained:
        print('Loading features for InceptionV3 model...')
    bottleneck_features = None
    try:
        bottleneck_features = np.load(BOTTLENECK_FEATURES)
    except FileNotFoundError:
        print('\nDownloading the DogInceptionV3Data.npz features...')
        download_bottleneck_url()
        bottleneck_features = np.load(BOTTLENECK_FEATURES)
    return bottleneck_features['train'], bottleneck_features['valid'], bottleneck_features['test']


# load list of dog names.
def load_dog_names():
    breeds = []
    with open(DOG_BREEDS, mode='r') as f:
        for line in f.readlines():
            breeds.append(line.replace('_', ' ').strip())
    return sorted(breeds)


def get_dataset():
    train_InceptionV3, valid_InceptionV3, test_InceptionV3 = load_bottleneck_features()

    # load train, test, and validation datasets.
    train_files, train_targets = load_dataset(os.path.join(DOG_IMAGES, 'train'))
    valid_files, valid_targets = load_dataset(os.path.join(DOG_IMAGES, 'valid'))
    test_files, test_targets = load_dataset(os.path.join(DOG_IMAGES, 'test'))

    dog_names = load_dog_names()

    # print statistics about the dataset.
    print('\nThere are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.\n' % len(test_files))
    return (train_InceptionV3, valid_InceptionV3, test_InceptionV3),\
           (train_files, train_targets), (valid_files, valid_targets),\
           (test_files, test_targets)


def load_model(path):
    model = create_model(load_bottleneck_features(model_is_trained=True)[0].shape[1:])
    print('Loading weights from %s for top layers...' % SAVED_MODEL)
    model.load_weights(path)
    return model


def create_model(pooling_shape):
    print('Creating and compiling a model...')
    model = keras.models.Sequential()
    model.add(keras.layers.GlobalAveragePooling2D(input_shape=pooling_shape))
    model.add(keras.layers.Dense(150, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(DOG_BREEDS_COUNT, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
