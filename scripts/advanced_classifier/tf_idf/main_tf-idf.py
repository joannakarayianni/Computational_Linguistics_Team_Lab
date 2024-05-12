#from naive_bayes import NaiveBayes
from tf_idf_embeddings import TFIDFVector
import numpy as np

def run_tf_idf(df):

    tfidf_vectorizer = TFIDFVector()

    if not df.empty:
        df = df.map(tfidf_vectorizer.preprocess_text)
        tfidf_vectorizer.calculate_idf(df.values.flatten())

    # Transforming texts to Tf-Idf vectors
    for row in df.iloc[:, 1:].values:
        flattened_row = [token for sublist in row for token in sublist]
        tfidf_vectorizer.caltulate_tfidf(flattened_row)

    # TODO: Not used
    max_length = max(len(vector) for vector in tfidf_vectorizer.tfidf_vectors)
    # Use padding
    padded_tfidf_vectors = []
    for vector in tfidf_vectorizer.tfidf_vectors:
        padded_vector = [vector.get(word, 0) for word in sorted(tfidf_vectorizer.idf_dict.keys())]
        padded_tfidf_vectors.append(padded_vector)
    tfidf_array = np.array(padded_tfidf_vectors)
    # Print Tf-Idf array
    print("TF-IDF:", tfidf_array)

if __name__ == "__main__":
    run_tf_idf()