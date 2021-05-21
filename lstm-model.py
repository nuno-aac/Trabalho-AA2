import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Bidirectional
import matplotlib.pyplot as plt


def get_first_dig(num):
    r = num.split('.')
    return int(r[0])


data = pd.read_pickle("data.pkl")

x = data['vectors'].to_numpy()
x = np.array(x.tolist())
x = x.astype('float32')


y = data['ec_number'].apply(get_first_dig).to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = models.Sequential()
model.add(Input(shape=(698, 100), dtype='float32', name='main_input'))
model.add(Bidirectional(LSTM(32,
                        return_sequences=True)))
model.add(Bidirectional(LSTM(32,
                        return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(8, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=64,
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