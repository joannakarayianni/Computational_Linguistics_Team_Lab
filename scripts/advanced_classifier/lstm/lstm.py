""" Run this file through the main.py """
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report

""" Long Short Memory class """
class LongShortTerm:
    
    def __init__(self, data_loader):
        self.df_train = data_loader.df_train
        self.df_val = data_loader.df_val
        self.df_test = data_loader.df_test
        self.y_train_labels = data_loader.y_train_labels
        self.y_val_labels = data_loader.y_val_labels
        self.y_test_labels = data_loader.y_test_labels

# Setting seeds for securing same results in every run
    @staticmethod
    def seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def train(self):
        # Setting the seeds we created
        self.seeds()

        # loading the training data, making sure they are strings 
        train_df = pd.read_csv('datasets/isear-train.csv', header=None, names=['emotion', 'text'], on_bad_lines='skip')
        train_df['text'] = train_df['text'].astype(str)

        # Making sure we use the emotion labels of the dataset (bc some lines have other words on the first column)
        emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']
        train_df = train_df[train_df['emotion'].isin(emotions)]
        
        # Tokenizing the training set
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_df['text'])
        train_sequences = tokenizer.texts_to_sequences(train_df['text'])
        word_index = tokenizer.word_index
       
        # Pading training sequences - all the same length
        max_sequence_length = 100
        X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)

        # Encoding the training labels
        mlb = MultiLabelBinarizer(classes=emotions)
        y_train = mlb.fit_transform(train_df['emotion'].apply(lambda x: [x]))

        # Loading the validation data, making sure they are strings
        val_df = pd.read_csv('datasets/isear-val.csv', header=None, names=['emotion', 'text'], on_bad_lines='skip')
        val_df['text'] = val_df['text'].astype(str)

        # Making sure we use the emotion labels of the dataset
        val_df = val_df[val_df['emotion'].isin(emotions)]

        # Tokenizing the validation text
        val_sequences = tokenizer.texts_to_sequences(val_df['text'])

        # Pading validation sequences
        X_val = pad_sequences(val_sequences, maxlen=max_sequence_length)

        # Encoding the validation labels
        y_val = mlb.transform(val_df['emotion'].apply(lambda x: [x]))

        # Setting vocabulary size and Parameters
        vocab_size = len(word_index) + 1
        embedding_dim = 100

        # Model Parameters
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dense(units=len(emotions), activation='sigmoid')) 
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Training 
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

        # Evaluating the model on validation set
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f'Validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')

        # Prediction on validation set
        y_pred = model.predict(X_val)
        y_pred = (y_pred > 0.5).astype(int)

        # Print Report
        print("Results for the validation set:")
        print(classification_report(y_val, y_pred, target_names=emotions))

        # Tokenizing the test text
        test_sequences = tokenizer.texts_to_sequences(self.df_test.iloc[:, 1])

        # Pading test sequences
        X_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

        # Encoding test labels
        y_test = mlb.transform(self.df_test.iloc[:, 0].apply(lambda x: [x]))

        # Evaluating model on test set
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')

        # Prediction on test set
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)
        predictions_df = pd.DataFrame(y_pred, columns=emotions)
        predictions_df.to_csv('predicions.csv', index=False)

        # Print Report
        print("Results for the test set:")
        print(classification_report(y_test, y_pred, target_names=emotions))