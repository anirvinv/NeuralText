import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

(train_X, train_y), (val_X, val_y) = mnist.load_data()
print('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
print('Test: X=%s, y=%s' % (val_X.shape, val_y.shape))

train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
val_X = val_X.reshape(val_X.shape[0], 28, 28, 1)

train_y = to_categorical(train_y)
val_y = to_categorical(val_y)

train_X = train_X.astype('float32')
val_X = val_X.astype('float32')


train_X = train_X/255.0
val_X = val_X/255.0


print(train_X.shape)

# cnn = tf.keras.Sequential([
#     layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_uniform", input_shape=(28,28,1)),
#     layers.MaxPooling2D((2,2)),

#     layers.Flatten(),
    
#     layers.Dense(100, activation="relu", kernel_initializer='he_uniform'),
#     layers.Dense(10, activation='softmax')
# ])

# cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# cnn.fit(train_X, train_y, epochs=1, batch_size=32, validation_data=(val_X, val_y))

# tf.keras.models.save_model(
#     cnn, "model", overwrite=True, include_optimizer=True, save_format=None,
#     signatures=None, options=None
# )