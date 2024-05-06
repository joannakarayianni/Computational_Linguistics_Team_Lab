"""Creating baseline classifier - MultiLabelPerceptron 
(for multilabel classification task - Emotion Recognition on text) 
Parameters: Training instances/ training data, learning rate- eta, labels (emotion labels-classes), iterations on training set
Attributes: weights, features"""

class MultiLabelPerceptron:
    def __init__(self, train_instances, labels, train_iterations=100, eta=0.1):
        self.train_instances = train_instances
        self.train_iterations = train_iterations
        self.labels = labels
        self.eta = eta
        
        # Attributes
        self.weights = {}  # Storing weights for each feature of each label
        self.initialize_weights()  # Calling function for weight initialization
        self.features = self.weights.keys()  # Set of all features with weights considered during training

    # Weight initializing function
    def initialize_weights(self):
        for label in self.labels:
            self.weights[label] = {}
            for train_instance in self.train_instances:
                feature_labels = train_instance.features
                for feature_label in feature_labels:
                    if feature_label not in self.weights[label].keys():
                        self.weights[label][feature_label] = 0.0

    # Perceptron training with training set
    def training_of_perceptron(self):
        for epoch in range(self.train_iterations):
            for train_instance in self.train_instances:
                features = train_instance.features 
                for label in self.labels:
                    # Correct label if it matches the current label
                    y_true = 1.0 if label in train_instance.emotions else -1.0
                    # Will be updated based on features and weights during prediction
                    y_predicted = self.predict(features, label) # We call predict function
                    if y_true != y_predicted:
                        self.update_weights(features, label, y_true) # We call update_weights function

    # Predicting the label for a given feature list
    def predict(self, features, label_to_predict):
        score = sum(self.weights[label_to_predict].get(feature, 0.0) for feature in features)
        return 1.0 if score >= 0 else -1.0

    # Update weights based on prediction error
    def update_weights(self, features, label, y_true):
        for feature_label in features:
            if feature_label in self.weights[label].keys():
                self.weights[label][feature_label] += self.eta * y_true
            else:
                self.weights[label][feature_label] = self.eta * y_true