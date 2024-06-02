import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

# Training data
train_df = pd.read_csv('isear-train.csv', header=None, names=['emotion', 'text'], on_bad_lines='skip')

# Ensure all sentences are strings
train_df['text'] = train_df['text'].astype(str)

# Emotions
emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']
train_df = train_df[train_df['emotion'].isin(emotions)]

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text'])

# Encode the training labels
mlb = MultiLabelBinarizer(classes=emotions)
y_train = mlb.fit_transform(train_df['emotion'].apply(lambda x: [x]))

# Validation data
val_df = pd.read_csv('isear-val.csv', header=None, names=['emotion', 'text'], on_bad_lines='skip')

# Ensure all sentences are strings
val_df['text'] = val_df['text'].astype(str)

# Include only the 7 emotions
val_df = val_df[val_df['emotion'].isin(emotions)]

# TF-IDF vectorization for validation data
X_val_tfidf = tfidf_vectorizer.transform(val_df['text'])

# Encode the validation labels
y_val = mlb.transform(val_df['emotion'].apply(lambda x: [x]))

# Define input shape
input_shape = X_train_tfidf.shape[1]

# Define input layer
input_layer = Input(shape=(input_shape,))

# Define hidden layers
hidden_layer = Dense(128, activation='relu')(input_layer)
dropout_layer = Dropout(0.5)(hidden_layer)

# Output layer
output_layer = Dense(len(emotions), activation='sigmoid')(dropout_layer)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
history = model.fit(X_train_tfidf, y_train, epochs=10, batch_size=32, validation_data=(X_val_tfidf, y_val))

# Evaluate the model on validation set
loss, accuracy = model.evaluate(X_val_tfidf, y_val, verbose=0)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Predict on validation set
y_pred = model.predict(X_val_tfidf)
y_pred = (y_pred > 0.5).astype(int)

# Report
print(classification_report(y_val, y_pred, target_names=emotions, zero_division=1))
