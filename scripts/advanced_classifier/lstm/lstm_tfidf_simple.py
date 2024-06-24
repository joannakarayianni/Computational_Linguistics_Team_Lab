import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

class LSTM_Tf_Idf:
    
    def __init__(self, train_path, val_path, seed=42):
        self.train_path = train_path
        self.val_path = val_path
        self.seed = seed
        self.emotions = ['joy', 'sadness', 'guilt', 'disgust', 'shame', 'fear', 'anger']
        self.seeds()
        self.load_data()
        self.prepare_data()
    
    def seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
    
    def load_data(self):
        # Load training data
        self.train_df = pd.read_csv(self.train_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        self.train_df['text'] = self.train_df['text'].astype(str)
        self.train_df = self.train_df[self.train_df['emotion'].isin(self.emotions)]

        # Load validation data
        self.val_df = pd.read_csv(self.val_path, header=None, names=['emotion', 'text'], on_bad_lines='skip')
        self.val_df['text'] = self.val_df['text'].astype(str)
        self.val_df = self.val_df[self.val_df['emotion'].isin(self.emotions)]
    
    def prepare_data(self):
        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.train_df['text'])
        self.X_val_tfidf = self.tfidf_vectorizer.transform(self.val_df['text'])

        # Encode labels
        self.mlb = MultiLabelBinarizer(classes=self.emotions)
        self.y_train = self.mlb.fit_transform(self.train_df['emotion'].apply(lambda x: [x]))
        self.y_val = self.mlb.transform(self.val_df['emotion'].apply(lambda x: [x]))
    
    def build_model(self):
        input_shape = self.X_train_tfidf.shape[1]
        input_layer = Input(shape=(input_shape,))
        hidden_layer = Dense(128, activation='relu')(input_layer)
        dropout_layer = Dropout(0.5)(hidden_layer)
        output_layer = Dense(len(self.emotions), activation='sigmoid')(dropout_layer)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
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
        
        print(classification_report(self.y_val, y_pred, target_names=self.emotions, zero_division=1))

# Example usage
data_loader = LSTM_Tf_Idf(train_path='datasets/isear-train.csv', val_path='datasets/isear-val.csv')
data_loader.build_model()
data_loader.train(epochs=10, batch_size=32) 

