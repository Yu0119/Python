# -*- coding: utf-8 -*-
"""
 Basic keras mnist training.
"""
# Import libs
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

# Consts
BATCHSIZE = 128
NCLASS = 10
EPOCHS = 30
NUM_TRAIN = 60000
NUM_TEST  = 10000
NROW, NCOL = 28, 28
F_MAX_UINT8 = 255.

# Load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data
x_train = x_train.reshape(NUM_TRAIN, NROW, NCOL, 1)
x_test  = x_test.reshape(NUM_TEST, NROW, NCOL, 1)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

# Normalize
x_train /= F_MAX_UINT8
x_test  /= F_MAX_UINT8

# To one-hot vector
y_train = keras.utils.to_categorical(y_train, NCLASS)
y_test  = keras.utils.to_categorical(y_test, NCLASS)

# Create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=(NROW, NCOL, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NCLASS, activation='softmax'))

# Show model
model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer=RMSprop(),
  metrics=['accuracy']
)

# Train
model.fit(
  x_train, y_train,
  batch_size=BATCHSIZE,
  epochs=EPOCHS,
  verbose=1,
  validation_data=(x_test, y_test)
)

# Evaluation
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
