""" Comparing the predictions of the 5 models: LSTM with word embeddings, Bi-LSTM with word embeddigns, LSTM with BERT embeddings,
LSTM with Tf-Idf embeddings & LSTM with GloVe embeddings, (predictions were on the test data) with the golden standard (test sataset with correct 
labels on each instance ) 
The metrics used where F1, Precision, Recall, F1 micro, F1 macro, Precision micro, recall micro and hamming loss
To directly run the file use --- python3 -m scripts.advanced_classifier.lstm.vaulting_lstm
"""
import pandas as pd
from scripts.evaluation.vaulting import Vaulting


# Loading prediction files for all the models
lstm_word_embeddings = pd.read_csv('scripts/advanced_classifier/lstm/predictions.csv')
bi_lstm_word_embeddings = pd.read_csv('scripts/advanced_classifier/lstm/predictionsbilstm.csv')
lstm_bert = pd.read_csv('scripts/advanced_classifier/lstm/predictionsbert.csv')
lstm_tfidf = pd.read_csv('scripts/advanced_classifier/lstm/predictionstfidf.csv')
lstm_glove = pd.read_csv('scripts/advanced_classifier/lstm/predictionsglove.csv')

# Golden standard (test data)
ground_truth_test = pd.read_csv('scripts/evaluation/ground_truths/ground_truth_test.csv')

# Models list
models = {
    'LSTM Word Embeddings': lstm_word_embeddings,
    'Bi-LSTM Word Embeddings': bi_lstm_word_embeddings,
    'LSTM BERT': lstm_bert,
    'LSTM TF-IDF': lstm_tfidf,
    'LSTM GloVe': lstm_glove
}


vaulting_instance = Vaulting(ground_truth_test, models)
vaulting_instance.findResults("lstm")
