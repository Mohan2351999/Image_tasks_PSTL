import h5py 
from functools import reduce
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import tensorflow
import random

from extra_keras_datasets import stl10
from keras.utils.np_utils import to_categorical

def load_stl10():
  #Load the STL dataset
  (x_train, y_train), (x_test, y_test) = stl10.load_data()

  # Preprocess the data - Normalize data
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  y_train = to_categorical(y_train, num_classes=10)
  y_test = to_categorical(y_test, num_classes=10)

  return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_stl10()

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

model.load_weights('/home/mohan235/projects/def-guzdial/mohan235/1_Mohan_GRAF_Work/Image_tasks_PSTL/Scratch_Training/cifar10_scratch_train.h5', by_name=True, skip_mismatch=True)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

score = model.evaluate(x_test, y_test)

print("The testing score on the model", score)
## Search Transfer mutation functions. - Modify here

# Add mutation operations - to a whole layer.
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Conv2D

def mutation_1(model):
    # Create a new model by cloning the original model
    new_model = clone_model(model)

    # Choose a random Conv2D layer to modify
    conv_layers = [layer for layer in new_model.layers if isinstance(layer, Conv2D)]
    conv_layer = np.random.choice(conv_layers)
    
    weights, biases = conv_layer.get_weights()

    random_tensor = np.random.rand(*weights.shape)

    modified_weights = weights + random_tensor

    conv_layer.set_weights([modified_weights, biases])

    opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    new_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return new_model

def mutation_2(model):
    # Create a new model by cloning the original model
    new_model = clone_model(model)

    # Choose a random Conv2D layer to modify
    conv_layers = [layer for layer in new_model.layers if isinstance(layer, Conv2D)]
    conv_layer = np.random.choice(conv_layers)
    
    weights, biases = conv_layer.get_weights()

    random_tensor = np.random.rand(*weights.shape)

    modified_weights = weights - random_tensor

    conv_layer.set_weights([modified_weights, biases])

    opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    new_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return new_model

def mutation_3(model):
    # Create a new model by cloning the original model
    new_model = clone_model(model)

    # Choose a random Conv2D layer to modify
    conv_layers = [layer for layer in new_model.layers if isinstance(layer, Conv2D)]
    conv_layer = np.random.choice(conv_layers)
    
    weights, biases = conv_layer.get_weights()

    random_tensor = np.random.rand(*weights.shape)

    modified_weights = weights * random_tensor

    conv_layer.set_weights([modified_weights, biases])

    opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    new_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return new_model

def mutation_4(model):
    # Create a new model by cloning the original model
    new_model = clone_model(model)

    # Choose a random Conv2D layer to modify
    conv_layers = [layer for layer in new_model.layers if isinstance(layer, Conv2D)]
    conv_layer = np.random.choice(conv_layers)
    
    weights, biases = conv_layer.get_weights()

    random_tensor = np.random.rand(*weights.shape)

    modified_weights = weights / random_tensor

    conv_layer.set_weights([modified_weights, biases])

    opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    new_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return new_model

def mutations(network):
    x = random.randint(0, 3)
    print(f">>>>>>>>>> Choose the mutation {x}")
    if x ==0:
        # Add weights to a layer.
        network = mutation_1(network)
    elif x == 1:
        # Subtract weights to a layer.
        network = mutation_2(network)    
    elif x== 2:
        # Multiply weights to a layer.
        network = mutation_3(network)
    elif x== 3:
        # Divide weights to a layer.
        network = mutation_4(network)
    
    return network

# Returns the loss on the testing data.
def fitness(model):
    return model.evaluate(x_train, y_train, verbose=2)[0]

def test_loss(model):
    return model.evaluate(x_test, y_test, verbose=2)[0]

import os
iterations = 100
neighbours_count = 10
min_loss = 10e6
np.random.seed(42)

base_model = model
for i in range(iterations):
    # Generate 10 neighbours
    print(f"*************   Started the Iteration {i}")  
    models = [base_model]
    for j in range(0, neighbours_count-1):
        models.append(mutations(base_model))
    for index, model_x in enumerate(models):
        if fitness(model_x) < min_loss:
            min_loss = fitness(model_x)
            base_model = model_x
    print(f"The minimum train loss in iteration {i}:", min_loss, " The test loss is ", test_loss(base_model))

print("Saved the model!!")
base_model.save("search_transfer_cifar10_to_stl10.h5")


score = base_model.evaluate(x_test, y_test)
print("The evaluation score (search transfer is): ", score)


### Assuming no data augmentation required...
base_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True)

base_model.save("search_transfer+backprop_cifar10_to_stl10.h5")
print("Saved the model after performing the search transfer+ backprop")
after_backprop_score = base_model.evaluate(x_test, y_test)
print("The evaluation score(search transfer and backprop is):", after_backprop_score)