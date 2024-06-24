import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load training data
train_df = pd.read_csv('isear-train.csv', header=None, names=['emotion', 'text'], on_bad_lines='skip')

# Ensure all sentences are strings
train_df['text'] = train_df['text'].astype(str)

# use the 7 emotions
emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']
train_df = train_df[train_df['emotion'].isin(emotions)]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
word_index = tokenizer.word_index

# Pad training sequences
max_sequence_length = 100
X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)

# Encode training labels
mlb = MultiLabelBinarizer(classes=emotions)
y_train = mlb.fit_transform(train_df['emotion'].apply(lambda x: [x]))

# Load validation data
val_df = pd.read_csv('isear-val.csv', header=None, names=['emotion', 'text'], on_bad_lines='skip')

# Ensure all sentences are strings
val_df['text'] = val_df['text'].astype(str)

# use the 7 emotions
val_df = val_df[val_df['emotion'].isin(emotions)]

# Tokenize validation text
val_sequences = tokenizer.texts_to_sequences(val_df['text'])

# Pad validation sequences
X_val = pad_sequences(val_sequences, maxlen=max_sequence_length)

# validation labels
y_val = mlb.transform(val_df['emotion'].apply(lambda x: [x]))

# Load GloVe embeddings, 100 dimensions
embedding_dim = 100
embedding_index = {}
with open(f'glove.6B.{embedding_dim}d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefficients

# embedding matrix
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=len(emotions), activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train 
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate model on validation set
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Predict on validation set
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)

# Classification report
print(classification_report(y_val, y_pred, target_names=emotions))

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axvline(x=len(history.history['loss']) - early_stopping.patience, color='r', linestyle='--', label='Early Stopping')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with Early Stopping')
plt.legend()
plt.show()