from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import advanced_classifier.word_embeddings.retrain_word2vec as CustomisedEmbeddings 

class SequentialNN:

    def __init__(self, df_train):
        self.df_train = df_train

    def train(self):
        # Generate word embeddings
        embedding_model = CustomisedEmbeddings(self.df_train)
        word2vec_model = embedding_model.generate_embeddings()
        # Define and compile neural network model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_embeddings.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_embeddings, y_train_labels, epochs=10, batch_size=32, validation_split=0.2)