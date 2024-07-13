import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report
from advanced_classifier.word_embeddings.custom_glove import CustomGloVe

class SequentialNNGlove:

    def __init__(self, data_loader, embedding_path):
        self.df_train = data_loader.df_train
        self.df_val = data_loader.df_val
        self.df_test = data_loader.df_test
        self.y_train_labels = data_loader.y_train_labels
        self.y_val_labels = data_loader.y_val_labels
        self.y_test_labels = data_loader.y_test_labels
        self.embedding_path = embedding_path

    def train(self):
        # Fetch word embeddings and labels

        # For training dataset
        embedding_model = CustomGloVe(self.embedding_path)
        X_train_embeddings = embedding_model.get_embeddings_matrix(self.df_train)
        
        # For validation dataset
        X_val_embeddings = embedding_model.get_embeddings_matrix(self.df_val)

        # Define and compile neural network model
        model = Sequential()

        # Emotions
        emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']

        # Define labels (emotions)
        y_train_labels_binary = pd.get_dummies(self.y_train_labels).values
        y_val_labels_binary = pd.get_dummies(self.y_val_labels).values

        # Add layers to the model
        model.add(Dense(128, activation='relu', input_shape=(X_train_embeddings.shape[1],)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(len(emotions), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_embeddings, y_train_labels_binary, epochs=30, batch_size=32, validation_data=(X_val_embeddings, y_val_labels_binary))

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
