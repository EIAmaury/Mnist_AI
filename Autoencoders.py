# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:55:24 2024

@author: amaur
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.models import Sequential

# data extraction
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

flatten=Flatten(dtype='float32')
#%% raw data
# data normalisation and organization
NOISE_FACTOR=0.5 # facteur de bruitage gaussian
x_train = x_train / 255.0
x_test=x_test/255
X_train_noisy = x_train + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
x_image_train=tf.reshape(x_train,[-1,28,28,1])
x_image_train=tf.cast(x_image_train,'float32')
x_image_test=tf.reshape(x_test,[-1,28,28,1])
x_image_test=tf.cast(x_image_test,'float32')

X_test_noisy = x_test + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

x_image_train_bruited=tf.reshape(X_train_noisy,[-1,28,28,1])
x_image_train_bruited=tf.cast(x_image_train_bruited,'float32')

x_image_test_bruited=tf.reshape(X_test_noisy,[-1,28,28,1])
x_image_test_bruited=tf.cast(x_image_test_bruited,'float32')
sample_example=5
fig,ax=plt.subplots(2,sample_example)
# presentation of data
for i in range(sample_example):
    ax[0][i].imshow(np.reshape(x_image_train[i],((28,28))))
    ax[0][i].set_title("Original")
    ax[0][i].axis('off')
    ax[1][i].imshow(np.reshape(x_image_train_bruited[i],((28,28))))
    ax[1][i].set_title("Bruited")
    ax[1][i].axis('off')
plt.show()

training_epochs = 2
           
#%% Autoencoders

#Encoding data
model=Sequential([
    Conv2D(128, (3, 3), activation='relu',padding='same', input_shape=(28,28,1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu',padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu',padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu',padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(1, (3, 3), activation='sigmoid',padding='same'),
    ])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
BATCH_SIZE=128
model.fit(x_image_train_bruited,x_image_train,epochs=training_epochs,batch_size=BATCH_SIZE,shuffle=True)
predictions=model.predict(x_image_test_bruited)
#► Result
examples_to_show=5
f, a = plt.subplots(2, examples_to_show, figsize=(12, 6))

for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(x_image_test_bruited[i], (28, 28)),cmap='gray')
    a[0][i].set_title("Original")
    a[0][i].axis('off')
    a[1][i].imshow(np.reshape(predictions[i], (28, 28)),cmap='gray')
    a[1][i].set_title("Reconstructed")
    a[1][i].axis('off')
    
plt.show()

#%% CNN for classification
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

classification=Sequential([
    Conv2D(64, (3, 3), activation='relu',padding='same', input_shape=(28,28,1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu',padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(1024,activation='relu'),
    Dense(10, activation='softmax')
    ])
classification.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
classification.fit(x_image_train,y_train,epochs=training_epochs,batch_size=BATCH_SIZE,shuffle=True)
predictions_class=classification.predict(predictions)

scores = classification.evaluate(predictions, y_test)
print("Neural network accuracy: %.2f%%" % (scores[1]*100))