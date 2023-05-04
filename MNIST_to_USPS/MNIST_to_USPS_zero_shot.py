import h5py 
from functools import reduce
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import tensorflow
import random

def hdf5(path, data_key = "data", target_key = "target", flatten = True):
    """
        loads data from hdf5: 
        - hdf5 should have 'train' and 'test' groups 
        - each group should have 'data' and 'target' dataset or spcify the key
        - flatten means to flatten images N * (C * H * W) as N * D array
    """
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get(data_key)[:]
        y_tr = train.get(target_key)[:]
        test = hf.get('test')
        X_te = test.get(data_key)[:]
        y_te = test.get(target_key)[:]
        if flatten:
            X_tr = X_tr.reshape(X_tr.shape[0], reduce(lambda a, b: a * b, X_tr.shape[1:]))
            X_te = X_te.reshape(X_te.shape[0], reduce(lambda a, b: a * b, X_te.shape[1:]))
    return X_tr, y_tr, X_te, y_te


import numpy as np
from keras.utils.np_utils import to_categorical

def get_usps():

    nb_classes = 10
    batch_size = 32
    input_shape = (16,16,1)


    # x_train, y_train, x_test, y_test = arrange_dataset_class_label(0)
    # x_train = x_train.reshape(50000, 32, 32, 3)
    # x_test = x_test.reshape(10000, 32, 32, 3)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255

    # y_train = to_categorical(y_train, nb_classes)
    # y_test = to_categorical(y_test, nb_classes)

    #########################  Change the data
    
    X_train, y_train, X_test, y_test = hdf5("usps.h5")
    X_train = X_train.reshape(X_train.shape[0], 16, 16, 1)
    X_test = X_test.reshape(X_test.shape[0], 16, 16, 1)

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (batch_size,  X_train, X_test, y_train, y_test)

batch_size, x_train, x_test, y_train, y_test = get_usps()

### Train validation split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

### Load the model(Cifar-Net)

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import os

batch_size = 32
num_classes = 10
epochs = 100
# data_augmentation = True


# Set random seed for Numpy
np.random.seed(10)
tf.random.set_seed(10)

## Define your model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.load_weights('/home/mohan235/projects/def-guzdial/mohan235/1_Mohan_GRAF_Work/Image_tasks_PSTL/Scratch_Training/mnist_scratch_train.h5', by_name=True, skip_mismatch=True)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# ### Assuming no data augmentation required...
# model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(x_val, y_val),
#               shuffle=True)


# model_json = model.to_json()
# with open("mnist_to_usps_finetuning_train.json", "w") as json_file:
#     json_file.write(model_json)

# serialize weights to HDF5
# model.save("mnist_to_usps_finetuning_train.h5")
# print("Saved model to disk!!!!")

score = model.evaluate(x_test, y_test)
print("The evaluation score is(zero-shot): ", score)