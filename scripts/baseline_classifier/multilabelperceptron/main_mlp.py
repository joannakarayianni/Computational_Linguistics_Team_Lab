from baseline_classifier.multilabelperceptron.emotion import EmotionSample
from baseline_classifier.multilabelperceptron.multilabelperceptron import MultiLabelPerceptron
import evaluation.evaluation_metrics as eval

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
    mlp_instance = MultiLabelPerceptron(train_instances=data, labels=labels, train_iterations=10, eta=0.1)

    # Training
    mlp_instance.training_of_perceptron()
    
    print("\nEvaluation Report :::")
    evaluate_mlp(df_test, mlp_instance)

def evaluate_mlp(df_test, mlp_instance):

    ####################### Test the classifier ####################### 
    labels = ['joy', 'anger', 'fear', 'sadness', 'disgust','guilt','shame']
    predicted_labels = []
    true_labels = []

    for _, row in df_test.iterrows():
        # unpacking the row fields.
        emotion_class, text = row
                    
        actual_label = emotion_class
        true_labels.append(actual_label)

        # Make a prediction using your classifier
        test_sample = EmotionSample([], text)
        max_score = -float('inf')
        predicted_label = None
        for label in labels:
            prediction = mlp_instance.predict(test_sample.features, label)
            if prediction > max_score:
                max_score = prediction
                predicted_label = label

        predicted_labels.append(predicted_label)
    
    eval.test_evaluation(true_labels, predicted_labels, labels)

if __name__ == "__main__":
    run_perceptron()