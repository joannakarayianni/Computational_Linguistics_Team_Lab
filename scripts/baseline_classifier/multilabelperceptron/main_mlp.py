import csv
from baseline_classifier.multilabelperceptron.emotion import EmotionSample
from baseline_classifier.multilabelperceptron.multilabelperceptron import MultiLabelPerceptron
import evaluation.evaluation_metrics as eval

def run_perceptron(df_train, df_test):

    mlp = None

    # Load data 
    def load_data_from_csv(file_path):
        samples = []
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader, None)  # Get headers
            if headers is not None:
                for row in reader:
                    if len(row) >= 2:  # Ensure there are at least two columns in the row
                        emotions = row[0].split()  # Emotions are space-separated in the first column
                        text = row[1]     # Text is in the second column
                        samples.append(EmotionSample(emotions, text))
        return samples 
    data = load_data_from_csv('datasets/isear-train.csv')

    # Labels
    labels = ['joy', 'anger', 'fear', 'sadness', 'disgust','guilt','shame']

    # MultiLabelPerceptron instance
    mlp = MultiLabelPerceptron(train_instances=data, labels=labels, train_iterations=10, eta=0.1)

    # Training
    mlp.training_of_perceptron()

    def evaluate_mlp():

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
                prediction = mlp.predict(test_sample.features, label)
                if prediction > max_score:
                    max_score = prediction
                    predicted_label = label

            predicted_labels.append(predicted_label)
        
        eval.test_evaluation(true_labels, predicted_labels, labels)

    evaluate_mlp()

if __name__ == "__main__":
    run_perceptron()