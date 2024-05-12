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




