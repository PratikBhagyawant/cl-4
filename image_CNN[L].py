import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize images (0-255 → 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

def build_model(learning_rate=0.001, 
                optimizer_name='adam', 
                dropout_rate=0.5, 
                num_filters=32, 
                kernel_size=(3,3)):

    model = models.Sequential()

    model.add(layers.Conv2D(num_filters, kernel_size, 
                            activation='relu', 
                            padding='same',
                            input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(num_filters*2, kernel_size, 
                            activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(num_filters*4, kernel_size, 
                            activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

model1 = build_model(learning_rate=0.001,
                     optimizer_name='adam',
                     dropout_rate=0.5,
                     num_filters=32,
                     kernel_size=(3,3))

history1 = model1.fit(X_train, y_train,
                      epochs=10,
                      batch_size=64,
                      validation_split=0.2)


model2 = build_model(learning_rate=0.01,
                     optimizer_name='sgd',
                     dropout_rate=0.3,
                     num_filters=64,
                     kernel_size=(5,5))

history2 = model2.fit(X_train, y_train,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)


test_loss, test_acc = model1.evaluate(X_test, y_test)
print("Test Accuracy (Model 1):", test_acc)

test_loss2, test_acc2 = model2.evaluate(X_test, y_test)
print("Test Accuracy (Model 2):", test_acc2)

plt.plot(history1.history['accuracy'], label='Train Acc (Model1)')
plt.plot(history1.history['val_accuracy'], label='Val Acc (Model1)')
plt.legend()
plt.show()