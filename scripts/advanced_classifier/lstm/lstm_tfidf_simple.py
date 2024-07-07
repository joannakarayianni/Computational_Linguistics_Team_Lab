""" Run this file through the main.py """
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

""" LSTM trained on Tf - Idf embeddings """
class LSTM_Tf_Idf:
    
    def __init__(self, train_path, val_path, test_path, seed=42):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.seed = seed
        self.emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']

        # Functions created
        self.seeds()
        self.load_data()
        self.prepare_data()
        self.build_model()
        self.evaluate_test_set()

# Setting seeds for securing same results in every run
    @staticmethod
    def seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

# Loading all the data in one function
    def load_data(self):
        # Loading training data, make sure they are strings, making sure they use emotion labels
        self.train_df = pd.read_csv(self.train_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        self.train_df['text'] = self.train_df['text'].astype(str)
        self.train_df = self.train_df[self.train_df['emotion'].isin(self.emotions)]

        # Loading validation data, make sure they are strings, making sure they use emotion labels
        self.val_df = pd.read_csv(self.val_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        self.val_df['text'] = self.val_df['text'].astype(str)
        self.val_df = self.val_df[self.val_df['emotion'].isin(self.emotions)]
        
        # Loading test data, make sure they are strings, making sure they use emotion labels
        self.test_df = pd.read_csv(self.test_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        self.test_df['text'] = self.test_df['text'].astype(str)
        self.test_df = self.test_df[self.test_df['emotion'].isin(self.emotions)]

 # Transforming instances to Tf-Idf vectors
    def prepare_data(self):
        # Tf-Idf vectorization
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.train_df['text']).toarray()
        self.X_val_tfidf = self.tfidf_vectorizer.transform(self.val_df['text']).toarray()
        self.X_test_tfidf = self.tfidf_vectorizer.transform(self.test_df['text']).toarray()

        # Encoding labels for all datasets
        self.mlb = MultiLabelBinarizer(classes=self.emotions)
        self.y_train = self.mlb.fit_transform(self.train_df['emotion'].apply(lambda x: [x]))
        self.y_val = self.mlb.transform(self.val_df['emotion'].apply(lambda x: [x]))
        self.y_test = self.mlb.transform(self.test_df['emotion'].apply(lambda x: [x]))
    
# Model Parameters
    def build_model(self):
        input_shape = self.X_train_tfidf.shape[1]
        input_layer = Input(shape=(input_shape,))
        hidden_layer = Dense(128, activation='relu')(input_layer)
        dropout_layer = Dropout(0.5)(hidden_layer)
        output_layer = Dense(len(self.emotions), activation='sigmoid')(dropout_layer)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
# Training
    def train(self, epochs=10, batch_size=32):
        self.model.fit(
            self.X_train_tfidf, self.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.X_val_tfidf, self.y_val)
        )
        
        loss, accuracy = self.model.evaluate(self.X_val_tfidf, self.y_val, verbose=0)
        print(f'Validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')
        
        y_pred = self.model.predict(self.X_val_tfidf)
        y_pred = (y_pred > 0.5).astype(int)

        # Print report
        print("Results for the validation set:")
        print(classification_report(self.y_val, y_pred, target_names=self.emotions, zero_division=1))
    
    # Evaluating the model on the test set
    def evaluate_test_set(self):
        loss, accuracy = self.model.evaluate(self.X_test_tfidf, self.y_test, verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')

        # Prediction on test set
        y_pred = self.model.predict(self.X_test_tfidf)
        y_pred = (y_pred > 0.5).astype(int)
        predictions_df = pd.DataFrame(y_pred, columns= self.emotions)
        predictions_df.to_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/predictionstfidf.csv', index=False)

        # Print Report
        print("Results for the test set:")
        print(classification_report(self.y_test, y_pred, target_names=self.emotions, zero_division=1))