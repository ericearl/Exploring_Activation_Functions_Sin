from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Conv2D, Input
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import simplejson as json

'''
This class is used to build the architecture of the models I want to test
then saves them in one of the test folders without testing them. I will then
go through all the architectures in all the test folders and actually fit the
models and evaluate them.
'''

# Define the sin activation function
def sin(x):
    return K.sin(x)

save_path = './tests/'
test_folder = 'test_9'
activations = ['relu', 'sigmoid', 'tanh', sin]
history = {}
num_epochs = 20
for i in range(len(activations)):
    name = activations[i] if i != len(activations)-1 else 'sin'

    # Define the model architecture
    input = Input(shape=(784,))
    x = Dense(units=5, activation=activations[i])(input)
    x = Dense(units=5, activation=activations[i])(x)
    x = Dense(units=5, activation=activations[i])(x)
    x = Dense(units=10, activation='softmax')(x)

    # Create the model and compile
    model = Model(inputs=input, outputs=x)

    # Save the model
    fp = save_path+test_folder+'/model_architecture_' + name + '.txt'
    with open(fp, 'w') as f:
        f.write(json.dumps(json.loads(model.to_json()),
                            indent=4,
                            sort_keys=True))
