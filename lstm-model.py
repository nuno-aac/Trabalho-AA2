import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Bidirectional, Dropout, Flatten
import matplotlib.pyplot as plt
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)
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


data = pd.read_pickle("parsed_data/data.pkl")

x = data['vectors'].tolist()
x = np.array(x, dtype='float32')


y = data['ec_number'].to_numpy()

y = y.reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = models.Sequential()
model.add(Input(shape=(698, 100), dtype='float32', name='main_input'))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit(x_train, y_train,
                    epochs=60,
                    batch_size=128,
                    validation_split=0.2)

history_dict = history.history

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

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