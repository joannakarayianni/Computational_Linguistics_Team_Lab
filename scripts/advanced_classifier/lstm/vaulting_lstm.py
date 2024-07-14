""" Comparing the predictions of the 5 models: LSTM with word embeddings, Bi-LSTM with word embeddigns, LSTM with BERT embeddings,
LSTM with Tf-Idf embeddings & LSTM with GloVe embeddings, (predictions were on the test data) with the golden standard (test sataset with correct 
labels on each instance ) 
The metrics used where F1, Precision, Recall, F1 micro, F1 macro, Precision micro, recall micro and hamming loss"""
import pandas as pd
from scripts.evaluation.vaulting import Vaulting
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


# Loading prediction files for all the models
lstm_word_embeddings = pd.read_csv('scripts/advanced_classifier/lstm/predicions.csv')
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


# vaulting_instance = Vaulting(ground_truth_test, models)
# vaulting_instance.findResults("lstm")

#Comment everything below once the above works.
# Storing performance metrics
metrics = pd.DataFrame(columns=['Model', 'Emotion', 'Hamming Loss', 'F1 Score', 'Precision', 'Recall', 'F1 Micro', 'F1 Macro', 'Precision Micro', 'Recall Micro'])

# Metrics for each model and each emotion separately
for model_name, predictions in models.items():
        for emotion in ground_truth_test.columns:  # Iterating over all columns
            
            y_true = ground_truth_test[emotion]
            y_pred = predictions[emotion]

            # Using average='binary' focuses the metric calculations on the positive class in binary classification tasks
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            ham_loss = hamming_loss(y_true, y_pred)
            f1_micro = f1_score(y_true, y_pred, average='micro')
            f1_macro = f1_score(y_true, y_pred, average='macro')
            precision_micro = precision_score(y_true, y_pred, average='micro')
            recall_micro = recall_score(y_true, y_pred, average='micro')

            metrics = metrics._append({
                'Model': model_name,
                'Emotion': emotion,
                'Hamming Loss': ham_loss,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'F1 Micro': f1_micro,
                'F1 Macro': f1_macro,
                'Precision Micro': precision_micro,
                'Recall Micro': recall_micro
            }, ignore_index=True)


# Saving the metrics 
metrics.to_csv('scripts/evaluation/all_results/evaluation_metrics.csv', index=False)

# Visualization for each emotion
for emotion in ground_truth_test.columns:
    
    emotion_metrics = metrics[metrics['Emotion'] == emotion]
    
    # Bar plot for Hamming Loss
    emotion_metrics.plot(x='Model', y='Hamming Loss', kind='bar', title=f'Hamming Loss for {emotion} by Model', legend=False)
    plt.ylabel('Hamming Loss')
    plt.show()

    # Bar plot for Precision
    emotion_metrics.plot(x='Model', y='Precision', kind='bar', title=f'Precision {emotion} by Model', legend=False)
    plt.ylabel('Precision')
    plt.show()

    # Bar plot for Recall
    emotion_metrics.plot(x='Model', y='Recall', kind='bar', title=f'Recall {emotion} by Model', legend=False)
    plt.ylabel('Recall')
    plt.show()

    # Bar plot for F1 score
    emotion_metrics.plot(x='Model', y='F1 Score', kind='bar', title=f'F1 score {emotion} by Model', legend=False)
    plt.ylabel('F1 score')
    plt.show()

    # Bar plot for F1  Micro and Macro Scores
    emotion_metrics.plot(x='Model', y=['F1 Micro', 'F1 Macro'], kind='bar', title=f'F1 Micro and Macro Scores for {emotion} by Model')
    plt.ylabel('F1 Micro & Macro Scores')
    plt.show()

    # Bar plot for Precision and Recall Micros
    emotion_metrics.plot(x='Model', y=['Precision Micro', 'Recall Micro'], kind='bar', title=f'Precision Micro and Recall Micro for {emotion} by Model')
    plt.ylabel('Precision Micro and Recall Micro')
    plt.show()