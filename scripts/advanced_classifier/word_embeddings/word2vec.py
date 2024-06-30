import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

class MyWord2Vec:

    def __init__(self, df_train):
        self.df_train = df_train
        self.tokenized_data = self.__tokenize_data__(self.df_train)
        self.word2vec_model = self.__train_word2vec__()
        
    def __tokenize_data__(self, df):
        return [word_tokenize(str(row[1].lower())) for _, row in df.iterrows()]
    
    def __train_word2vec__(self):
        # Define Word2Vec model parameters
        vector_size = 100  # Dimensionality of the word vectors
        window = 5  # Maximum distance between the current and predicted word within a sentence
        min_count = 1  # Ignores all words with a total frequency lower than this
        workers = 4  # Number of CPU cores to use for training

        # Train Word2Vec model
        word2vec_model = Word2Vec(sentences=self.tokenized_data,
                                  vector_size=vector_size,
                                  window=window,
                                  min_count=min_count,
                                  workers=workers)
        return word2vec_model

    def get_embeddings_matrix(self, df):
        tokenized_data = self.__tokenize_data__(df)
        # Create embedding matrix
        embedding_dim = self.word2vec_model.vector_size
        embedding_matrix = np.zeros((len(tokenized_data), embedding_dim))

        # Iterate over each tokenized text
        for i, tokens in enumerate(tokenized_data):
            # Calculate mean embedding for each text
            embeddings = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
            if embeddings:
                mean_embedding = np.mean(embeddings, axis=0)
                embedding_matrix[i] = mean_embedding
            else:
                # Handle missing embeddings by using a default value (zeros)
                default_embedding = np.zeros(embedding_dim)
                embedding_matrix[i] = default_embedding
        
        return embedding_matrix

# Example of how to use the CustomWord2Vec class
# Assuming df_train is your training DataFrame
# custom_w2v = CustomWord2Vec(df_train)
# X_train_embeddings = custom_w2v.get_embeddings_matrix()
