import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

class WordEmbeddingTrainer:

    def __init__(self, df_train):
        self.df_train = df_train
        
    def __generate_embeddings__(self):
        # Load emotion-labeled text data
        

        # Tokenize the text data.
        # We train the model only on the text data and not the labels. 
        # The word2vec accordingly generates embeddings.
        # These embeddings are vectors of a given dimension which capture different aspects of the word.
        # Words with similar meanings or contexts will have similar vector representations.
        # The distance between these vectors in the embedding space reflects their similarity.
        tokenized_data = [word_tokenize(row[1].lower()) for _, row in self.df_train.iterrows()]

        # Define Word2Vec model parameters
        vector_size = 100  # Dimensionality of the word vectors
        window = 5  # Maximum distance between the current and predicted word within a sentence
        min_count = 1  # Ignores all words with a total frequency lower than this
        workers = 4  # Number of CPU cores to use for training

        # Train Word2Vec model
        word2vec_model = Word2Vec(sentences=tokenized_data,
                        vector_size=vector_size,
                        window=window,
                        min_count=min_count,
                        workers=workers)
        # Save the trained Word2Vec model
        word2vec_model.save("emotion_word2vec.model")

    def get_embeddings_matrix(self):
        # TODO: Make tokenized_data into a helper function
        tokenized_data = [word_tokenize(row[1].lower()) for _, row in self.df_train.iterrows()]

        # Load pre-trained Word2Vec model
        word2vec_model = Word2Vec.load("emotion_word2vec.model")
        
        # Create embedding matrix
        embedding_dim = word2vec_model.vector_size
        embedding_matrix = np.zeros((len(tokenized_data), embedding_dim))

        # Iterate over each tokenized text
        for i, tokens in enumerate(tokenized_data):
            # Calculate mean embedding for each text
            embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
            if embeddings:
                mean_embedding = np.mean(embeddings, axis=0)
                embedding_matrix[i] = mean_embedding

        return embedding_matrix, self.df_train.iloc[:, 0]
    