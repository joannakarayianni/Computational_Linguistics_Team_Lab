""" Run this file through the main.py 
Also, in order to run this we need the 100 dimentions gloVe file which I downloaded from this website: https://nlp.stanford.edu/projects/glove/ , 'glove.6B.zip'
and attached it to the same folder as this script """

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

""" LSTM trained on GloVe embeddings with 100 dimensions """

class LSTM_GloVe:
    def __init__(self, data_loader, glove_path='scripts/advanced_classifier/lstm/glove.6B.100d.txt'):
        self.data_loader = data_loader
        self.glove_path = glove_path
        self.emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']

        # Functions created
        self.seeds()
        self.prepare_data()
        self.build_model()
        self.train()
        self.evaluate_test_set()

    # Setting seeds for securing same results in every run
    @staticmethod
    def seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def prepare_data(self):
        # Tokenizing training set
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.data_loader.df_train.iloc[:, 1].astype(str))
        train_sequences = tokenizer.texts_to_sequences(self.data_loader.df_train.iloc[:, 1].astype(str))
        word_index = tokenizer.word_index

        # Padding sequences - all the same length
        max_sequence_length = 100
        self.X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)

        # Encoding the training labels
        mlb = MultiLabelBinarizer(classes=self.emotions)
        self.y_train = mlb.fit_transform(self.data_loader.y_train_labels.apply(lambda x: [x]))

        # Tokenizing validation set & test set
        val_sequences = tokenizer.texts_to_sequences(self.data_loader.df_val.iloc[:, 1].astype(str))
        self.X_val = pad_sequences(val_sequences, maxlen=max_sequence_length)
        test_sequences = tokenizer.texts_to_sequences(self.data_loader.df_test.iloc[:, 1].astype(str))
        self.X_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

        # Encoding the validation and test labels
        self.y_val = mlb.transform(self.data_loader.y_val_labels.apply(lambda x: [x]))
        self.y_test = mlb.transform(self.data_loader.y_test_labels.apply(lambda x: [x]))

        # Vocabulary size and embedding dimensions / Parameters
        self.vocab_size = len(word_index) + 1
        self.embedding_dim = 100

        # Loading the GloVe embeddings
        self.embedding_matrix = self.load_glove_embeddings(self.glove_path, word_index, self.embedding_dim)

#Open and load GloVe embeddings file
    def load_glove_embeddings(self, glove_file, word_index, embedding_dim=100):
        embeddings_index = {}
        with open(glove_file, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix
    
# Model Parameters
    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.X_train.shape[1],
                            weights=[self.embedding_matrix], trainable=False))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))
        model.add(Dense(units=len(self.emotions), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model
# Training 
    def train(self, epochs=10, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_val, self.y_val))

        # Evaluating model on validation set
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        print(f'Validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')

        # Prediction on validation set
        y_pred = self.model.predict(self.X_val)
        y_pred = (y_pred > 0.5).astype(int)

        # Print report
        print("Results for the validation set:")
        print(classification_report(self.y_val, y_pred, target_names=self.emotions))

    # Evaluating model on test set
    def evaluate_test_set(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')

        # Prediction on test set
        y_pred = self.model.predict(self.X_test)
        y_pred = (y_pred > 0.5).astype(int)
        predictions_df = pd.DataFrame(y_pred, columns= self.emotions)
        predictions_df.to_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/predictionsglove.csv', index=False)

        # Print report
        print("Results for the test set:")
        print(classification_report(self.y_test, y_pred, target_names=self.emotions))