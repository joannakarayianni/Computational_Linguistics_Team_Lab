""" Comparing the predictions of the 5 models: LSTM with word embeddings, Bi-LSTM with word embeddigns, LSTM with BERT embeddings,
LSTM with Tf-Idf embeddings & LSTM with GloVe embeddings, (predictions were on the test data) with the golden standard (test sataset with correct 
labels on each instance ) 
The metrics used where F1 micro, F1 macro, Precision micro, recal micro and hamming loss"""
import pandas as pd
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Loading prediction files for all the models
lstm_word_embeddings = pd.read_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/predicions.csv')
bi_lstm_word_embeddings = pd.read_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/predictionsbilstm.csv')
lstm_bert = pd.read_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/predictionsbert.csv')
lstm_tfidf = pd.read_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/predictionstfidf.csv')
lstm_glove = pd.read_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/predictionsglove.csv')

# Golden standard (test data)
ground_truth = pd.read_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/ground_truth.csv')

# Models list
models = {
    'LSTM Word Embeddings': lstm_word_embeddings,
    'Bi-LSTM Word Embeddings': bi_lstm_word_embeddings,
    'LSTM BERT': lstm_bert,
    'LSTM TF-IDF': lstm_tfidf,
    'LSTM GloVe': lstm_glove
}

# Storing performance metrics
metrics = pd.DataFrame(columns=['Model', 'Emotion', 'Hamming Loss', 'F1 Micro', 'F1 Macro', 'Precision Micro', 'Recall Micro'])

# Metrics for each model and each emotion separately
for model_name, predictions in models.items():
    for emotion in ground_truth.columns:  # Iterating over all columns
        if emotion == 'index':  
            continue
        
        y_true = ground_truth[emotion]
        y_pred = predictions[emotion]

        # Dropping rows with NaN values
        non_nan_indices = y_pred.dropna().index
        y_true = y_true.loc[non_nan_indices]
        y_pred = y_pred.loc[non_nan_indices]

        ham_loss = hamming_loss(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        
        metrics = metrics._append({
            'Model': model_name,
            'Emotion': emotion,
            'Hamming Loss': ham_loss,
            'F1 Micro': f1_micro,
            'F1 Macro': f1_macro,
            'Precision Micro': precision_micro,
            'Recall Micro': recall_micro
        }, ignore_index=True)

# Saving the metrics 
metrics.to_csv('/Users/ioannakaragianni/Documents/GitHub/Computational_Linguistics_Team_Lab/scripts/advanced_classifier/lstm/metrics.csv', index=False)


# Visualization for each emotion
for emotion in ground_truth.columns:
    if emotion == 'index':  # Skip the index column assuming it's not an emotion
        continue
    
    emotion_metrics = metrics[metrics['Emotion'] == emotion]
    
    # Bar plot for Hamming Loss
    emotion_metrics.plot(x='Model', y='Hamming Loss', kind='bar', title=f'Hamming Loss for {emotion} by Model', legend=False)
    plt.ylabel('Hamming Loss')
    plt.show()

    # Bar plot for F1 Scores
    emotion_metrics.plot(x='Model', y=['F1 Micro', 'F1 Macro'], kind='bar', title=f'F1 Scores for {emotion} by Model')
    plt.ylabel('F1 Score')
    plt.show()

    # Bar plot for Precision and Recall
    emotion_metrics.plot(x='Model', y=['Precision Micro', 'Recall Micro'], kind='bar', title=f'Precision and Recall for {emotion} by Model')
    plt.ylabel('Score')
    plt.show()