import h5py 
from functools import reduce

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

batch_size, x_train, x_test, y_train, y_test = get_usps()

### Train validation split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

### Load the model(Cifar-Net)

from __future__ import print_function
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

# Convert class vectors to binary class matrices.
train_labels = keras.utils.to_categorical(train_labels, num_classes)


# Set random seed for Numpy
np.random.seed(10)
tf.random.set_seed(10)

## Define your model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_data.shape[1:]))
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
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.load('mnist_scratch_train.h5')

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

## Search Transfer mutation functions. - Modify here

# Add mutation operations - to a whole layer.
def mutation_1(network):
    # Change the weights of a particular conv2d layer.
    conv_locations = []

    for i, layer in enumerate(network.layers):
        # print(i, layer, layer.name)
        if(isinstance(layer, keras.layers.Conv2DTranspose)):
        conv_locations.append(i)
        # print("Conv2DTranspose layer weights shape: ", len(layer.get_weights()), layer.get_weights()[0].shape, layer.get_weights()[1].shape)

    # print("All the conv layers:", conv_locations)
    # Randomly pick a location.
    random_conv_location = random.choice(conv_locations)
    for i, layer in enumerate(network.layers):
        if i== random_conv_location:
        generated_weights = np.random.uniform(0, 1, size=layer.get_weights()[0].shape)
        new_weights = np.add(layer.get_weights()[0], generated_weights)
        assign_weights = [new_weights, layer.get_weights()[1]] # Assigning both weights and bias.
        network.layers[i].set_weights(assign_weights)
        print(f"Added weights to a Conv2d layer at index {random_conv_location}")

    # sys.exit("Exiting program here")
    return network

# Subtract mutation operations - to a whole layer.
def mutation_2(network):
    # Change the weights of a particular conv2d layer.
    conv_locations = []

    for i, layer in enumerate(network.layers):
        # print(i, layer, layer.name)
        if(isinstance(layer, keras.layers.Conv2DTranspose)):
        conv_locations.append(i)
        # print("Conv2DTranspose layer weights shape: ", len(layer.get_weights()), layer.get_weights()[0].shape, layer.get_weights()[1].shape)

    # print("All the conv layers:", conv_locations)
    # Randomly pick a location.
    random_conv_location = random.choice(conv_locations)
    for i, layer in enumerate(network.layers):
        if i== random_conv_location:
        generated_weights = np.random.uniform(0, 1, size=layer.get_weights()[0].shape)
        new_weights = np.subtract(layer.get_weights()[0], generated_weights)
        assign_weights = [new_weights, layer.get_weights()[1]] # Assigning both weights and bias.
        network.layers[i].set_weights(assign_weights)
        print(f"Subtract weights to a Conv2d layer at index {random_conv_location}")

    # sys.exit("Exiting program here")
    return network

# Multiply mutation operations - to a whole layer.
def mutation_3(network):
    # Change the weights of a particular conv2d layer.
    conv_locations = []

    for i, layer in enumerate(network.layers):
        # print(i, layer, layer.name)
        if(isinstance(layer, keras.layers.Conv2DTranspose)):
        conv_locations.append(i)
        # print("Conv2DTranspose layer weights shape: ", len(layer.get_weights()), layer.get_weights()[0].shape, layer.get_weights()[1].shape)

    # print("All the conv layers:", conv_locations)
    # Randomly pick a location.
    random_conv_location = random.choice(conv_locations)
    for i, layer in enumerate(network.layers):
        if i== random_conv_location:
        generated_weights = np.random.uniform(0, 1, size=layer.get_weights()[0].shape)
        new_weights = np.multiply(layer.get_weights()[0], generated_weights)
        assign_weights = [new_weights, layer.get_weights()[1]] # Assigning both weights and bias.
        network.layers[i].set_weights(assign_weights)
        print(f"Multiply weights to a Conv2d layer at index {random_conv_location}")

    # sys.exit("Exiting program here")
    return network

# Divide mutation operations - to a whole layer.
def mutation_4(network):
    # Change the weights of a particular conv2d layer.
    conv_locations = []

    for i, layer in enumerate(network.layers):
        # print(i, layer, layer.name)
        if(isinstance(layer, keras.layers.Conv2DTranspose)):
        conv_locations.append(i)
        # print("Conv2DTranspose layer weights shape: ", len(layer.get_weights()), layer.get_weights()[0].shape, layer.get_weights()[1].shape)

    # print("All the conv layers:", conv_locations)
    # Randomly pick a location.
    random_conv_location = random.choice(conv_locations)
    for i, layer in enumerate(network.layers):
        if i== random_conv_location:
        generated_weights = np.random.uniform(0, 1, size=layer.get_weights()[0].shape)
        new_weights = np.divide(layer.get_weights()[0], generated_weights)
        assign_weights = [new_weights, layer.get_weights()[1]] # Assigning both weights and bias.
        network.layers[i].set_weights(assign_weights)
        print(f"Divide weights to a Conv2d layer at index {random_conv_location}")

    # sys.exit("Exiting program here")
    return network

def mutations(network):
    x = random.randint(0, 5)
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
    return model.evaluate(x_train, y_train, verbose=2)

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
base_model.save("search_transfer_mnist_to_usps.h5")

score = model.evaluate(x_test, y_test)
print("The evaluation score is: ", score)