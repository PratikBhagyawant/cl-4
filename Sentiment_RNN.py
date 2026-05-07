import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import networkx as nx

df = pd.read_csv("Tweets.csv")

df.shape

df.head()

df.isnull().sum()

df.dropna(subset=['text','sentiment'], inplace = True)

texts = df['text'].values
texts

sentiments = df['sentiment'].values
sentiments

le = LabelEncoder()

labels = le.fit_transform(sentiments)
print(labels)

y= to_categorical(labels)

# Step 4: Text Preprocessing
tokenizer = Tokenizer(num_words = 1000, oov_token='<OOV>')

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

x = pad_sequences(sequences, maxlen= 100)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim = 1000,output_dim=2, input_length=100),
    SimpleRNN(32),
    Dense(3, activation= 'softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

pred_probs = model.predict(x[:200])  # Limit to 200 for clarity in graph
pred_labels = np.argmax(pred_probs, axis=1)

pred_sentiments = le.inverse_transform(pred_labels)

# Sample 10 tweets
sample_tweets = df['text'].values[:10]
sample_sequences = x[:10]

preds = np.argmax(model.predict(sample_sequences), axis=1)
pred_sentiments = le.inverse_transform(preds)

# Sample 10 tweets
sample_tweets = df['text'].values[:10]
sample_sequences = x[:10]

# Predict sentiments
preds = np.argmax(model.predict(sample_sequences), axis=1)
pred_sentiments = le.inverse_transform(preds)

# Build Network Graph
G = nx.Graph()

# Add nodes with sentiment
for i, text in enumerate(sample_tweets):
    label = f"Tweet {i+1}\nSentiment: {pred_sentiments[i]}"
    G.add_node(i, label=label, sentiment=pred_sentiments[i])

# Connect nodes with same predicted sentiment
for i in range(len(preds)):
    for j in range(i+1, len(preds)):
        if preds[i] == preds[j]:
            G.add_edge(i, j)

# Visualize graph
pos = nx.spring_layout(G, seed=42)
colors = ['green' if G.nodes[n]['sentiment'] == 'positive'
          else 'red' if G.nodes[n]['sentiment'] == 'negative'
          else 'gray' for n in G.nodes()]
labels = nx.get_node_attributes(G, 'label')

# plt.figure(figsize=(12, 8))
# nx.draw(G, pos, with_labels=True, labels=labels,
#         node_color=colors, node_size=1800, font_size=9)
# plt.title("Tweet Sentiment Network Graph (RNN)", fontsize=14)
# plt.axis('off')
# plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
nx.draw(G, pos, ax=ax, with_labels=True, labels=labels,
        node_color=colors, node_size=1800, font_size=9)
plt.title("Tweet Sentiment Network Graph (RNN)", fontsize=14)
plt.axis('off')
plt.show()