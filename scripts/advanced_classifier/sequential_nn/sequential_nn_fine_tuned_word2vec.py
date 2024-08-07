import numpy as np
import pandas as pd
from keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from advanced_classifier.word_embeddings.custom_word2vec import CustomWord2Vec
from sklearn.metrics import classification_report

class SequentialNNWithFineTunedW2Vec:

    def __init__(self, data_loader):
        self.df_train = data_loader.df_train
        self.df_val = data_loader.df_val
        self.df_test = data_loader.df_test
        self.y_train_labels = data_loader.y_train_labels
        self.y_val_labels = data_loader.y_val_labels
        self.y_test_labels = data_loader.y_test_labels

    def train(self):

        # Fetch word embeddings and labels

        # For training dataset
        # This step creates the fine-tuned word2vec model - emotion_word2vec.model
        embedding_model = CustomWord2Vec(self.df_train)
        X_train_embeddings = embedding_model.get_embeddings_matrix(self.df_train)
        
        # For validation dataset
        X_val_embeddings = embedding_model.get_embeddings_matrix(self.df_val)

        # Define and compile neural network model
        # Define the Sequential model
        model = Sequential()

        # emotions
        emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']

        # Define labels (emotions)
        y_train_labels_binary = pd.get_dummies(self.y_train_labels).values
        y_val_labels_binary = pd.get_dummies(self.y_val_labels).values

        # Add layers to the model one by one
        model.add(Dense(64, activation='relu', input_shape=(X_train_embeddings.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_embeddings, y_train_labels_binary, epochs=15, batch_size=32, validation_data=(X_val_embeddings, y_val_labels_binary))

        # Evaluate the model on validation set
        loss, accuracy = model.evaluate(X_val_embeddings, y_val_labels_binary, verbose=0)
        print(f'Validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')

        # Predict on validation set
        y_pred = model.predict(X_val_embeddings)
        
        y_pred_binary = np.argmax(y_pred, axis=1)
        y_val_binary = np.argmax(y_val_labels_binary, axis=1)


        # Report
        print(classification_report(y_val_binary, y_pred_binary, target_names=emotions))