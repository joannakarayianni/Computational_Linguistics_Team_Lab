import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

class CustomWord2Vec:

    def __init__(self, df_train):
        self.df_train = df_train
        self.__train_word2vec__()
        
    def __train_word2vec__(self):
        # Load emotion-labeled text data
        

        # Tokenize the text data.
        # We train the model only on the text data and not the labels. 
        # The word2vec accordingly generates embeddings.
        # These embeddings are vectors of a given dimension which capture different aspects of the word.
        # Words with similar meanings or contexts will have similar vector representations.
        # The distance between these vectors in the embedding space reflects their similarity.
        tokenized_data = [word_tokenize(str(row[1].lower())) for _, row in self.df_train.iterrows()]

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

    def get_embeddings_matrix(self, df):
        tokenized_data = []

        for index, row in df.iterrows():
            try:
                tokenized_text = word_tokenize(str(row[1]).lower())
                tokenized_data.append(tokenized_text)
            except AttributeError as e:
                print(f"Error processing row at index {index}: {row}")
                raise e
        
        # Load pre-trained Word2Vec model
        word2vec_model = Word2Vec.load("emotion_word2vec.model")
        
        # Create embedding matrix
        embedding_dim = word2vec_model.vector_size
        embedding_matrix = np.zeros((len(tokenized_data), embedding_dim))

        # Iterate over each tokenized text
        for i, tokens in enumerate(tokenized_data):
            # Calculate mean embedding for each text
            # length of embeddings would be equal to the number of tokens in text/row.
            # each elememt within embeddings is again a vector of length vector_size
            embeddings = []
            for word in tokens:
                if word in word2vec_model.wv:
                    embeddings.append(word2vec_model.wv[word])
            if embeddings:
                # length of mean_embedding is same as the vector_size
                mean_embedding = np.mean(embeddings, axis=0)
                # mean embedding is just one flattened vector formed by mean of all the vectors representing the text.
                embedding_matrix[i] = mean_embedding
            else:
                # Handle missing embeddings by using a default value (zeros)
                word_embedding_dim = word2vec_model.vector_size
                default_embedding = np.zeros(word_embedding_dim)
                # length of mean_embedding is same as the vector_size
                print(default_embedding, len(default_embedding))
                embedding_matrix[i] = default_embedding
        
        return embedding_matrix
    