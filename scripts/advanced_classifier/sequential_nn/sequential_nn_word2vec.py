import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from advanced_classifier.word_embeddings.word2vec import Word2VecWrapper
from sklearn.metrics import classification_report

class SequentialNNWord2Vec:

    def __init__(self, data_loader):
        self.df_train = data_loader.df_train
        self.df_val = data_loader.df_val
        self.df_test = data_loader.df_test
        self.y_train_labels = data_loader.y_train_labels
        self.y_val_labels = data_loader.y_val_labels
        self.y_test_labels = data_loader.y_test_labels
        self.emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']
        self.seeds()
        self.build_model()
    
    # Setting seeds for securing same results in every run
    @staticmethod
    def seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    def build_model(self):
        self.model = Sequential()
        self.embedding_model = Word2VecWrapper(self.df_train)
    
    def train_test(self):
        self.train()
        self.test()

    def train(self):

        # Fetch word embeddings and labels

        # For training dataset
        X_train_embeddings = self.embedding_model.get_embeddings_matrix(self.df_train)
        
        # For validation dataset
        X_val_embeddings = self.embedding_model.get_embeddings_matrix(self.df_val)

        # Define labels (emotions)
        y_train_labels_binary = pd.get_dummies(self.y_train_labels).values
        y_val_labels_binary = pd.get_dummies(self.y_val_labels).values

        # Add layers to the model one by one
        self.model.add(Dense(64, activation='relu', input_shape=(X_train_embeddings.shape[1],)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(7, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(X_train_embeddings, y_train_labels_binary, epochs=15, batch_size=32, validation_data=(X_val_embeddings, y_val_labels_binary))

        # Evaluate the model on validation set
        loss, accuracy = self.model.evaluate(X_val_embeddings, y_val_labels_binary, verbose=0)
        print(f'Validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')

        # Predict on validation set
        y_pred = self.model.predict(X_val_embeddings)
        
        y_pred_binary = np.argmax(y_pred, axis=1)
        y_val_binary = np.argmax(y_val_labels_binary, axis=1)


        # Report
        print(classification_report(y_val_binary, y_pred_binary, target_names=self.emotions))

    def test(self):
      
        # For test dataset
        X_test_embeddings = self.embedding_model.get_embeddings_matrix(self.df_test)

        # Define labels (emotions)
        y_test_labels_binary = pd.get_dummies(self.y_test_labels).values

        # Predict on test set
        y_pred = self.model.predict(X_test_embeddings)

         # Evaluate the model on validation set
        loss, accuracy = self.model.evaluate(X_test_embeddings, y_test_labels_binary, verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')
        
        # Convert predictions to binary format (one-hot encoded)
        y_pred_binary = np.zeros_like(y_pred)
        y_pred_binary[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1

        # Save to CSV
        pred_df = pd.DataFrame(y_pred_binary, columns=self.emotions)
        pred_df.to_csv('scripts/advanced_classifier/sequential_nn/predictions/predictions_seq_nn_word2vec.csv', index=False)

        # Convert test labels to one-hot for classification report
        y_test_labels = np.argmax(y_test_labels_binary, axis=1)

        # Report
        print(classification_report(y_test_labels, np.argmax(y_pred_binary, axis=1), target_names=self.emotions))