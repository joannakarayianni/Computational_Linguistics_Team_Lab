""" Creating baseline classifier - Perceptron class 
(for binary classification task) to help us with the multilabel perceptron 
Parameters: Training instances, learning rate- eta, labels (emotion labels-classes), iterations on training set
Attributes: weights, features """

class Perceptron:
    def __init__(self, train_instances, label, train_iterations=10, eta=0.1):
        self.train_instances = train_instances
        self.train_iterations = train_iterations
        self.label = label
        self.eta = eta
        
        # Attributes
        self.weights = {}  # Storing weights for each feature
        self.initialize_weights()  # Calling function for weight initialization

        self.features = self.weights.keys()  # Set of all features with weights considered during training

    # Weight initializing function
    def initialize_weights(self):
        for train_instance in self.train_instances + [self.label]:
            feature_labels = train_instance.features
            for feature_label in feature_labels:
                if feature_label not in self.weights.keys():
                    self.weights[feature_label] = 0.0

    # Perceptron training with provided instances
    def training_of_perceptron(self):
        for epoch in range(self.train_iterations):
            for train_instance in self.train_instances:
                features = train_instance.features 

                # Correct label if it matches the correct emotion
                y_true = 1.0 if train_instance.emotion == self.label.emotion else -1.0
                # Will be updated based on features and weights during prediction
                y_predicted = self.predict(features)

                if y_true != y_predicted:
                    self.update_weights(features, y_true)

    # Predicts the label for a given feature list
    # Returns a binary prediction
    def predict(self, features):
        score = sum(self.weights.get(feature, 0.0) for feature in features)
        return 1.0 if score >= 0 else -1.0

    # Update weights based on prediction error
    def update_weights(self, features, y_true):
        for feature_label in features:
            if feature_label in self.weights.keys():
                self.weights[feature_label] += self.eta * y_true
            else:
                self.weights[feature_label] = self.eta * y_true

# -------------------Testing--------------------
import csv

class EmotionSample:
    def __init__(self, emotion, text):
        self.emotion = emotion
        self.text = text
        self.features = self.extract_features()

    def extract_features(self):
        return self.text.split()

# Load data from CSV file
def load_data_from_csv(file_path):
    samples = []
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader, None)  # Get the headers
        if headers is not None:
            for row in reader:
                if len(row) >= 2:  # Ensure there are at least two columns in the row
                    emotion = row[0]  # Emotion is in the first column
                    text = row[1]     # Text is in the second column
                    samples.append(EmotionSample(emotion, text))
    return samples 

# Load data from CSV
data = load_data_from_csv('/Users/ioannakaragianni/Desktop/shit/isear-train.csv')

# Define the label
label = EmotionSample('joy','When I understood that I was admitted to the University')
# Create a Perceptron instance
perceptron = Perceptron(train_instances=data, label=label, train_iterations=10, eta=0.1)

# Train the perceptron
perceptron.training_of_perceptron()

# Test the perceptron with a new emotion sample
test_sample = EmotionSample('joy', 'I hate this weather') # Made up example
prediction = perceptron.predict(test_sample.features)

if prediction == 1.0:
    print("Predicted: Joy")
else:
    print("Predicted: Not Joy")