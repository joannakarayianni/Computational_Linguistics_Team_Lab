""" TF - IDF class for feature extraction on data set 
-Text preprocessing/ Normalisation 
- TF
- IDF
- TF - TFIDF array """ 

import pandas as pd
import string
import math
import numpy as np

# TF-IDF Vector class creation
class TFIDFVector:
    def __init__(self):
        self.idf_dict = {}
        self.tfidf_vectors = []
        self.tokenized_docs = []
# Text normalisation
    def preprocess_text(self, text):
        if text is None:
            return []
        # Tokenization
        tokens = text.split()
        # Lowercasing
        tokens = [word.lower() for word in tokens]
        # Removal of punctuation
        tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens]
        # Removal of empty tokens
        tokens = [word for word in tokens if word]
        return tokens
# Tf
    def calculate_tf(self, tokens):
        tf_dict = {}
        total_words = len(tokens)
        for word in tokens:
            tf_dict[word] = tf_dict.get(word, 0) + 1 / total_words
        return tf_dict
# Idf
    def calculate_idf(self, df):
        tokenized_documents = []
        if not df.empty:
            tokenized_documents = df.map(self.preprocess_text)
        
        self.tokenized_docs = tokenized_documents.values.flatten()

        total_docs = len(tokenized_documents)
        all_words = set([word for tokenized_doc in self.tokenized_docs for word in tokenized_doc])
        for word in all_words:
            doc_count = sum(1 for tokenized_doc in self.tokenized_docs if word in tokenized_doc)
            self.idf_dict[word] = math.log(total_docs / (1 + doc_count))
# Tf-Idf
    def caltulate_tfidf(self, text):
        tfidf_vector = {}
        tf_dict = self.calculate_tf(text)
        for word, tf in tf_dict.items():
            if word in self.idf_dict:
                tfidf_vector[word] = tf * self.idf_dict[word]
            else:
                # Update Idf dictionary for new words
                self.idf_dict[word] = math.log(len(self.idf_dict) + 1)  # smoothing
                tfidf_vector[word] = tf * self.idf_dict[word]
        self.tfidf_vectors.append(tfidf_vector)

#******************************* Testing **********************************************

# def load_data_from_csv(file_path):
#     try:
#         return pd.read_csv(file_path, header=None)
#     except pd.errors.ParserError: # some lines have more than 1 columns, ignore
#         print("ParserError: Some lines have inconsistent number of fields. Skipping those lines.")
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#         data = []
#         for line in lines:
#             fields = line.strip().split(',')
#             if len(fields) == 1:
#                 # skiping incompatible lines
#                 continue
#             data.append(fields)
#         return pd.DataFrame(data)

# # Load data and preprocess
# training_data = load_data_from_csv('./datasets/isear-train.csv')
# tfidf_vectorizer = TFIDFVector()
# if not training_data.empty:
#     training_data = training_data.map(tfidf_vectorizer.preprocess_text)
#     tfidf_vectorizer.calculate_idf(training_data.values.flatten())

# # Transforming texts to Tf-Idf vectors
# for row in training_data.iloc[:, 1:].values:
#     flattened_row = [token for sublist in row for token in sublist]
#     tfidf_vectorizer.caltulate_tfidf(flattened_row)

# print(tfidf_vectorizer.idf_dict)

# max_length = max(len(vector) for vector in tfidf_vectorizer.tfidf_vectors)
# # Use padding
# padded_tfidf_vectors = []
# for vector in tfidf_vectorizer.tfidf_vectors:
#     padded_vector = [vector.get(word, 0) for word in sorted(tfidf_vectorizer.idf_dict.keys())]
#     padded_tfidf_vectors.append(padded_vector)
# tfidf_array = np.array(padded_tfidf_vectors)
# Print Tf-Idf array
# print("TF-IDF:", tfidf_array)

# Checking if the vectors are all 0s 
# Iterate through each vector in the TF-IDF array
# for vector in tfidf_array:
#     # Check if any element in the vector is not equal to 0
#     if any(element != 0 for element in vector):
#        print("Non 0 elements found :", vector)
#     else:
#         print("All elements are 0:", vector)




