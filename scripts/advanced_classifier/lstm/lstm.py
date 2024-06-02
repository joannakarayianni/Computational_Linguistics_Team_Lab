import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report

# training data
train_df = pd.read_csv('isear-train.csv', header=None, names=['emotion', 'text'], on_bad_lines='skip')

# first few rows of training dataset
#print(train_df.head())

# ensure all sentences are strings
train_df['text'] = train_df['text'].astype(str)

# emotions
emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']
train_df = train_df[train_df['emotion'].isin(emotions)]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
word_index = tokenizer.word_index

# Pad training sequences to ensure they are all the same length
max_sequence_length = 100
X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)

# Encode the training labels
mlb = MultiLabelBinarizer(classes=emotions)
y_train = mlb.fit_transform(train_df['emotion'].apply(lambda x: [x]))

# validation data
val_df = pd.read_csv('isear-val.csv', header=None, names=['emotion', 'text'], on_bad_lines='skip')

# first rows of the validation data
#print(val_df.head())

# ensure all sentences are strings
val_df['text'] = val_df['text'].astype(str)

# include only the 7 emotions
val_df = val_df[val_df['emotion'].isin(emotions)]

# Tokenize the validation text
val_sequences = tokenizer.texts_to_sequences(val_df['text'])

# Pad validation sequences
X_val = pad_sequences(val_sequences, maxlen=max_sequence_length)

# Encode the validation labels
y_val = mlb.transform(val_df['emotion'].apply(lambda x: [x]))

# parameters
vocab_size = len(word_index) + 1
embedding_dim = 100

# LSTM 
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dense(units=len(emotions), activation='sigmoid'))

# Compile 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on validation set
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Predict on validation set
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)

# Report
print(classification_report(y_val, y_pred, target_names=emotions))

