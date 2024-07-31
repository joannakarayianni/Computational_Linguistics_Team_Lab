""" Comparing the predictions of the 5 models: LSTM with word embeddings, Bi-LSTM with word embeddigns, BERT,
LSTM with Tf-Idf embeddings & LSTM with GloVe embeddings, (predictions were on the test data) with the golden standard (test sataset with correct 
labels on each instance ) 
The metrics used where F1, Precision, Recall, F1 micro, F1 macro, Precision micro, recall micro and hamming loss
To directly run the file use --- python3 -m scripts.advanced_classifier.lstm.lstm_model_comparison_visualizer
"""
import pandas as pd
from scripts.evaluation.models_comparision_visualizer import ModelComparisonVisualizer


# Loading prediction files for all the models
lstm_word_embeddings = pd.read_csv('scripts/advanced_classifier/lstm/predictions/predictions.csv')
bi_lstm_word_embeddings = pd.read_csv('scripts/advanced_classifier/lstm/predictions/predictionsbilstm.csv')
lstm_bert = pd.read_csv('scripts/advanced_classifier/lstm/predictions/predictionsbert.csv')
lstm_tfidf = pd.read_csv('scripts/advanced_classifier/lstm/predictions/predictionstfidf.csv')
lstm_glove = pd.read_csv('scripts/advanced_classifier/lstm/predictions/predictionsglove.csv')

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


visualizer_instance = ModelComparisonVisualizer(ground_truth_test, models)
visualizer_instance.findResults("lstm")
