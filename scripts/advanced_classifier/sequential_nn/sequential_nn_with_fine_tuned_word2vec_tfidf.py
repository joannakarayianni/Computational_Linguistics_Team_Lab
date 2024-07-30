import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from advanced_classifier.word_embeddings.custom_word2vec import CustomWord2Vec
from sklearn.metrics import classification_report
from advanced_classifier.tf_idf.tf_idf_embeddings import TFIDFVector 
from gensim.models import Word2Vec


class SequentialNNCustomWord2VecTFIDF:

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
        return self.model
    
    def train_test(self):
        self.train()
        self.test()

    def train(self):

        # Fetch word embeddings and labels
    
        # For training dataset
        # This step creates the fine-tuned word2vec model - emotion_word2vec.model
        CustomWord2Vec(self.df_train)

        X_train_embeddings_with_tfidf = self.__combine_embeddings__(self.df_train)
        # For validation dataset
        X_val_embeddings_with_tfidf = self.__combine_embeddings__(self.df_val)

        # Define labels (emotions)
        y_train_labels_binary = pd.get_dummies(self.y_train_labels).values
        y_val_labels_binary = pd.get_dummies(self.y_val_labels).values

        
        # Add layers to the model one by one
        self.model.add(Dense(64, activation='relu', input_shape=(X_train_embeddings_with_tfidf.shape[1],)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(7, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(X_train_embeddings_with_tfidf, y_train_labels_binary, epochs=15, batch_size=32, validation_data=(X_val_embeddings_with_tfidf, y_val_labels_binary))

        # Evaluate the model on validation set
        loss, accuracy = self.model.evaluate(X_val_embeddings_with_tfidf, y_val_labels_binary, verbose=0)
        print(f'Validation Loss: {loss}')
        print(f'Validation Accuracy: {accuracy}')

        # Predict on validation set
        y_pred = self.model.predict(X_val_embeddings_with_tfidf)
        
        y_pred_binary = np.argmax(y_pred, axis=1)
        y_val_binary = np.argmax(y_val_labels_binary, axis=1)


        # Report
        print(classification_report(y_val_binary, y_pred_binary, target_names=self.emotions))
    
    def test(self):
      
        # For test dataset
        X_test_embeddings_with_tfidf = self.__combine_embeddings__(self.df_test)

        # Define labels (emotions)
        y_test_labels_binary = pd.get_dummies(self.y_test_labels).values

        # Predict on test set
        y_pred = self.model.predict(X_test_embeddings_with_tfidf)

        # Evaluate the model on validation set
        loss, accuracy = self.model.evaluate(X_test_embeddings_with_tfidf, y_test_labels_binary, verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')
        
        # Convert predictions to binary format (one-hot encoded)
        y_pred_binary = np.zeros_like(y_pred)
        y_pred_binary[np.arange(len(y_pred)), np.argmax(y_pred, axis=1)] = 1

        # Save to CSV
        pred_df = pd.DataFrame(y_pred_binary, columns=self.emotions)
        pred_df.to_csv('scripts/advanced_classifier/sequential_nn/predictions/predictions_seq_nn_word2vec_tfidf.csv', index=False)

        # Convert test labels to one-hot for classification report
        y_test_labels = np.argmax(y_test_labels_binary, axis=1)

        # Report
        print(classification_report(y_test_labels, np.argmax(y_pred_binary, axis=1), target_names=self.emotions))

    def __combine_embeddings__(self, df):
        word2vec_model = Word2Vec.load("emotion_word2vec.model")
        tfidf_vectorizer = TFIDFVector()
        tfidf_vectorizer.calculate_idf(df)

        all_tokenized_data = df.iloc[:, 1].map(tfidf_vectorizer.preprocess_text)
 
        for tokenized_data in all_tokenized_data:
            tfidf_vectorizer.caltulate_tfidf(tokenized_data)

        combined_embeddings = []

        for tfidf_vector in tfidf_vectorizer.tfidf_vectors:
            word_embeddings = []
            for token, tfidf_score in tfidf_vector.items():
                if token in word2vec_model.wv:
                    word_embedding = word2vec_model.wv[token]
                    combined_embedding = word_embedding * tfidf_score
                    # each of these combined embeddings is of word2vec_model vector_size.
                    word_embeddings.append(combined_embedding)
                else:
                    # Handle missing embeddings by using a default value (zeros)
                    word_embedding_dim = word2vec_model.vector_size
                    default_embedding = np.zeros(word_embedding_dim)
                    combined_embedding = default_embedding * tfidf_score
                    word_embeddings.append(combined_embedding)

            if word_embeddings:
                # combined_embedding_mean is flattened list of size vector_size
                combined_embedding_mean = np.mean(word_embeddings, axis=0)
                combined_embeddings.append(combined_embedding_mean)

        return np.array(combined_embeddings)
        

