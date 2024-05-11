import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import advanced_classifier.word_embeddings.custom_word2vec as w2v 


class SequentialNN:

    def __init__(self, df_train, df_val, df_test):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

    def train(self):

        # Fetch word embeddings and labels

        # For training dataset
        embedding_model = w2v.CustomWord2Vec(self.df_train)
        X_train_embeddings, y_train_labels = embedding_model.get_embeddings_matrix()
        
        # For validation dataset
        embedding_model_val = w2v.CustomWord2Vec(self.df_val)
        X_val_embeddings, y_val_labels = embedding_model_val.get_embeddings_matrix()

        # Define and compile neural network model
        # Define the Sequential model
        model = Sequential()

        # Define labels (emotions)
        y_train_labels_binary = pd.get_dummies(y_train_labels).values
        y_val_labels_binary = pd.get_dummies(y_val_labels).values

        # Add layers to the model one by one
        model.add(Dense(64, activation='relu', input_shape=(X_train_embeddings.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_embeddings, y_train_labels_binary, epochs=15, batch_size=32, validation_data=(X_val_embeddings, y_val_labels_binary))