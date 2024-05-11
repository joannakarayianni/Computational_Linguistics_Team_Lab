import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import advanced_classifier.word_embeddings.custom_word2vec as w2v
from advanced_classifier.tf_idf.tf_idf_embeddings import TFIDFVector 
from gensim.models import Word2Vec


class SequentialNNWithTFIDF:

    def __init__(self, df_train, df_val, df_test):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

    def train(self):

        # Fetch word embeddings and labels
    

        # For training dataset
        embedding_model = w2v.CustomWord2Vec(self.df_train)
        embedding_model.get_embeddings_matrix()
        X_train_embeddings, y_train_labels = embedding_model.get_embeddings_matrix()
        X_train_embeddings_with_tfidf = self.__combine_embeddings__(self.df_train)

        # For validation dataset
        embedding_model_val = w2v.CustomWord2Vec(self.df_val)
        X_val_embeddings, y_val_labels = embedding_model_val.get_embeddings_matrix()
        X_val_embeddings_with_tfidf = self.__combine_embeddings__(self.df_val)

        # Define and compile neural network model
        # Define the Sequential model
        model = Sequential()

        # Define labels (emotions)
        y_train_labels_binary = pd.get_dummies(y_train_labels).values
        y_val_labels_binary = pd.get_dummies(y_val_labels).values

        
        # Add layers to the model one by one
        model.add(Dense(64, activation='relu', input_shape=(X_train_embeddings_with_tfidf.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_embeddings, y_train_labels_binary, epochs=15, batch_size=32, validation_data=(X_val_embeddings_with_tfidf, y_val_labels_binary))

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
                    word_embeddings.append(combined_embedding)
            if word_embeddings:
                combined_embedding_mean = np.mean(word_embeddings, axis=0)
                combined_embeddings.append(combined_embedding_mean)

        
        return np.array(combined_embeddings)
        

