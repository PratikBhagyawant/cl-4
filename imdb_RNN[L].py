import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, GRU, Dense
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load IMDB dataset (Top 10000 words)
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

max_length = 200

X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

print("Shape after padding:", X_train.shape)

word_index = imdb.get_word_index()

# Add special tokens offset
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

def encode_review(text):
    tokens = text.lower().split()
    encoded = [1]  # start token
    
    for word in tokens:
        if word in word_index and word_index[word] < vocab_size:
            encoded.append(word_index[word])
        else:
            encoded.append(2)  # unknown word
    
    return pad_sequences([encoded], maxlen=max_length)

model = Sequential()

# Embedding Layer
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))

# LSTM Layer
model.add(LSTM(64))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

y_pred = (model.predict(X_test) > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample_review = "This movie was absolutely amazing and fantastic"
encoded_review = encode_review(sample_review)

prediction = model.predict(encoded_review)

if prediction > 0.5:
    print("Positive Review 😊")
else:
    print("Negative Review 😞")

model1 = Sequential()

# Embedding Layer
model1.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))

# GRU Layer
model1.add(GRU(64))

# Output Layer
model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model1.summary()

history = model1.fit(X_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.2)

loss, accuracy = model1.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

y_pred = (model1.predict(X_test) > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample_review = "The movie was boring and waste of time"
encoded_review = encode_review(sample_review)

prediction = model1.predict(encoded_review)

if prediction > 0.5:
    print("Positive Review 😊")
else:
    print("Negative Review 😞")

model2 = Sequential()

# Embedding Layer
model2.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))

# SimpleRNN Layer
model2.add(SimpleRNN(64))

# Output Layer
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.summary()

history = model2.fit(X_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.2)

loss, accuracy = model2.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

y_pred = (model2.predict(X_test) > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample_review = "This movie was very bad and disappointing"
encoded_review = encode_review(sample_review)

prediction = model2.predict(encoded_review)

if prediction > 0.5:
    print("Positive Review 😊")
else:
    print("Negative Review 😞")