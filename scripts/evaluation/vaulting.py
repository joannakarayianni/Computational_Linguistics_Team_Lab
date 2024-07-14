import pandas as pd
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

class Vaulting:

    def __init__(self, ground_truth, models):
        self.ground_truth = ground_truth
        self.models = models
    

    def findResults(self, seq_or_lstm):
        # Storing performance metrics
        metrics = pd.DataFrame(columns=['Model', 'Emotion', 'Hamming Loss', 'F1 Score', 'Precision', 'Recall', 'F1 Micro', 'F1 Macro', 'Precision Micro', 'Recall Micro'])

        # Metrics for each model and each emotion separately
        for model_name, predictions in self.models.items():
            for emotion in self.ground_truth.columns:  # Iterating over all columns
                
                y_true = self.ground_truth[emotion]
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
        metrics.to_csv(f'scripts/evaluation/all_results/{seq_or_lstm}-evaluation_metrics.csv', index=False)

        # Visualization for each emotion
        for emotion in self.ground_truth.columns:
            
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