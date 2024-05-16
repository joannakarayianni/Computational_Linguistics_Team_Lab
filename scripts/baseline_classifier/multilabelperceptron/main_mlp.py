from baseline_classifier.multilabelperceptron.emotion import EmotionSample
from baseline_classifier.multilabelperceptron.multilabelperceptron import MultiLabelPerceptron


def run_perceptron(df_train, df_test):
    print("Running Multilabel Perceptron algorithm")
    # Load data 
    data = []
    for _, row in df_train.iterrows():
        if len(row) >= 2:  # Ensure there are at least two columns in the row
            emotions = row[0].split()  # Emotions are space-separated in the first column
            text = row[1]     # Text is in the second column
            data.append(EmotionSample(emotions, text))
    
    # Labels
    labels = ['joy', 'anger', 'fear', 'sadness', 'disgust','guilt','shame']

    # MultiLabelPerceptron instance
    mlp_instance = MultiLabelPerceptron(df_test, train_instances=data, labels=labels, train_iterations=10, eta=0.1)

    # Training
    mlp_instance.training_of_perceptron()
    

if __name__ == "__main__":
    run_perceptron()