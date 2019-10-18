# Imports and what not
from keras import backend as K
from keras.models import Model
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

# Save information about this test
save_path = './tests/'
test_folder = 'test_1'

if not os.path.exists(save_path + test_folder):
    os.mkdir(save_path + test_folder)

'''
We are going to repeat the process of building the model using each activation
function and training the model. After this is done for all activation functions
we will compare the results.
'''
activations = ['relu', 'sigmoid', 'tanh', sin]
history = {}
num_epochs = 20
for i in range(len(activations)):
    name = activations[i] if i != len(activations)-1 else 'sin'

    # Define the model architecture
    input = Input(shape=(784,))
    x = Dense(units=100, activation=activations[i])(input)
    x = Dense(units=50, activation=activations[i])(x)
    x = Dense(units=10, activation='softmax')(x)

    # Create the model and compile
    model = Model(inputs=input, outputs=x)
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

    # Save the model
    fp = save_path+test_folder+'/model_architecture_' + name + '.txt'
    with open(fp, 'w') as f:
        f.write(json.dumps(json.loads(model.to_json()),
                            indent=4,
                            sort_keys=True))

df = pd.DataFrame(history)
df.to_csv(save_path+test_folder+'/training_history.csv')
