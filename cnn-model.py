import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, Flatten, MaxPooling1D, Dropout
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def get_first_dig(num):
    r = num.split('.')
    return int(r[0])


print('Reading')

data = pd.read_pickle("parsed_data/data.pkl")

print(data)

x = data['vectors'].to_numpy()

y = data['ec_number'].apply(get_first_dig).to_numpy()

print('Splitting data set')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = models.Sequential()
model.add(Input(shape=(698, 100), dtype='float32', name='main_input'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(8, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.summary()

print('EVAL')

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.2)

history_dict = history.history

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(acc_values) + 1)

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()


test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
