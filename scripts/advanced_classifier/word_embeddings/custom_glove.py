import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

class CustomGloVe:

    def __init__(self, embedding_path):
        self.embeddings_index = self.__load_glove_embeddings__(embedding_path)
        
    def __load_glove_embeddings__(self, embedding_path):
        embeddings_index = {}
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, "Reading GloVe embeddings"):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def get_embeddings_matrix(self, df):
        tokenized_data = []

        for index, row in df.iterrows():
            try:
                tokenized_text = word_tokenize(str(row[1]).lower())
                tokenized_data.append(tokenized_text)
            except AttributeError as e:
                print(f"Error processing row at index {index}: {row}")
                raise e
        
        embedding_dim = len(next(iter(self.embeddings_index.values())))
        embedding_matrix = np.zeros((len(tokenized_data), embedding_dim))

        for i, tokens in enumerate(tokenized_data):
            embeddings = []
            for word in tokens:
                if word in self.embeddings_index:
                    embeddings.append(self.embeddings_index[word])
            if embeddings:
                mean_embedding = np.mean(embeddings, axis=0)
                embedding_matrix[i] = mean_embedding
            else:
                default_embedding = np.zeros(embedding_dim)
                embedding_matrix[i] = default_embedding
        
        return embedding_matrix
