""" Comparing the predictions of the 3 best Seq. NN configurations: word2vec with dropout, word2vec with tf-idf, seq. nn with glove embeddings against the golden standard (test sataset with correct 
labels on each instance ) 
The metrics used where F1, Precision, Recall, F1 micro, F1 macro, Precision micro, recall micro and hamming loss
"""

import pandas as pd
from scripts.evaluation.models_comparision_visualizer import ModelComparisonVisualizer



# Loading prediction files for all the models
sequential_nn_word2vec_dropout = pd.read_csv('scripts/advanced_classifier/sequential_nn/predictions/predictions_seq_nn_word2vec_dropout.csv')
sequential_nn_word2vec_tfidf = pd.read_csv('scripts/advanced_classifier/sequential_nn/predictions/predictions_seq_nn_word2vec_tfidf.csv')
sequential_nn_glove = pd.read_csv('scripts/advanced_classifier/sequential_nn/predictions/predictions_seq_nn_glove.csv')

# Golden standard (test data)
ground_truth_test = pd.read_csv('scripts/evaluation/ground_truths/ground_truth_test.csv')

# Models list
models = {
    'Sequential NN with word2vec and dropout': sequential_nn_word2vec_dropout,
    'Sequential NN with word2vec and tf-idf': sequential_nn_word2vec_tfidf,
    'Sequential NN with Glove': sequential_nn_glove,
}


visualizer_instance = ModelComparisonVisualizer(ground_truth_test, models)
visualizer_instance.findResults("seq_nn")