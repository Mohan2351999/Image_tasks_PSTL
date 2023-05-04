from keras.datasets import mnist
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

## Load the dataset

def load_cifar10():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  # Preprocess the data
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  y_train = to_categorical(y_train, num_classes=10)
  y_test = to_categorical(y_test, num_classes=10)

  return x_train, y_train, y_train, y_test

x_train, x_test, y_train, y_test = load_cifar10()

### Train validation split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print("The trainig data shape: ", x_train.shape)
print("The training labels shape is: ",y_train.shape)
print("The testing images shape is: ",x_test.shape)
print("The testinglabels shape is: ",y_test.shape)

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

# Set random seed for TensorFlow (Keras backend)
tf.random.set_seed(10)

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

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

### Assuming no data augmentation required...
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True)


# model_json = model.to_json()
# with open("mnist_scratch_train.json", "w") as json_file:
#     json_file.write(model_json)

# serialize weights to HDF5
model.save("cifar10_scratch_train.h5")
print("Saved model to disk!!!!")


score = model.evaluate(x_test, y_test)
print("The evaluation score is(scratch training): ", score)