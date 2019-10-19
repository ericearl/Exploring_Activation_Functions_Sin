# Imports and what not
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Conv2D, Input
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import simplejson as json

'''
Here I am actually fitting the architectures and doing some tests on how each
model compares.
'''

# Load and Prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Get validation data for validating during training
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.1)

# Define the sin activation function
def sin(x):
    return K.sin(x)

get_custom_objects().update({'sin': Activation(sin)})


# Save information about this test
save_path = './tests/test_'
num_tests = 9
activations = ['relu', 'sigmoid', 'tanh', 'sin']
for i in range(1, num_tests+1):
    '''
    We are going to repeat the process of building the model using each
    activation function and training the model. After this is done for all
    activation functions we will compare the results.
    '''
    history = {}
    num_epochs = 100
    for j in range(len(activations)):
        name = activations[j]
        path = save_path + str(i) + '/model_architecture_' + name + '.txt'
        with open(path, 'r') as f:
            model = model_from_json(f.read())

            model.compile(optimizer='adam',
                            metrics=['accuracy'],
                            loss='mean_squared_error')

            # Fit the model
            history[name] = model.fit(x_train,
                                        y_train,
                                        validation_data=(x_val, y_val),
                                        epochs=num_epochs,
                                        batch_size=64,
                                        verbose=0).history

    df = pd.DataFrame(history)
    df.to_csv(save_path + str(i) + '/training_history.csv')
