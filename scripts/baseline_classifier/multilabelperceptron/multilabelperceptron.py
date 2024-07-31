"""Creating baseline classifier - MultiLabelPerceptron 
(for multilabel classification task - Emotion Recognition on text) 
Parameters: Training instances/ training data, learning rate- eta, labels (emotion labels-classes), iterations on training set
Attributes: weights, features"""

from baseline_classifier.multilabelperceptron.emotion import EmotionSample
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

class MultiLabelPerceptron:
    def __init__(self, dataloader, labels, train_iterations=100, eta=0.1):
        self.df_train = dataloader.df_train
        self.df_val = dataloader.df_val
        self.df_test = dataloader.df_test
        self.train_iterations = train_iterations
        self.labels = labels
        self.eta = eta
        
        # Attributes
        self.weights = {}  # Storing weights for each feature of each label
        self.initialize_weights()  # Calling function for weight initialization
        self.features = set(self.weights.keys())  # Set of all features with weights considered during training

    # Weight initializing function
    def initialize_weights(self):
        for label in self.labels:
            self.weights[label] = {}
            for _, row in self.df_train.iterrows():
                feature_labels = row[1].split()
                for feature_label in feature_labels:
                    if feature_label not in self.weights[label].keys():
                        self.weights[label][feature_label] = 0.0

    # Perceptron training with training set
    def training_of_perceptron(self):
        for epoch in range(self.train_iterations):
            train_correct = 0
            train_total = 0
            for _, row in self.df_train.iterrows():
                emotions = row[0].split()
                text = row[1]
                train_instance = EmotionSample(emotions, text)
                features = train_instance.features 
                for label in self.labels:
                    # Correct label if it matches the current label
                    y_true = 1.0 if label in train_instance.emotions else -1.0
                    # Will be updated based on features and weights during prediction
                    y_predicted = self.predict(features, label)  # We call predict function
                    if y_true == y_predicted:
                        train_correct += 1
                    train_total += 1
                    if y_true != y_predicted:
                        self.update_weights(features, label, y_true)  # We call update_weights function
            #train_accuracy = train_correct / train_total
            # Evaluating on the validation set
            self.evaluate_on_dev_dataset(epoch)
        # Evaluating on the test set after training completes
        self.evaluate_on_test_dataset()
# evaluation on the validation set
    def evaluate_on_dev_dataset(self, epoch):
        true_labels = []
        predicted_labels = []
        for _, row in self.df_val.iterrows():
            emotion_class, text = row
            actual_label = emotion_class
            true_labels.append(actual_label)

            # Making predictions using our classifier
            data_instance = EmotionSample([], text)
            max_score = -float('inf')
            predicted_label = None
            for label in self.labels:
                prediction = self.predict(data_instance.features, label)
                if prediction > max_score:
                    max_score = prediction
                    predicted_label = label
            predicted_labels.append(predicted_label)

        # Computing evaluation metrics (accuracy, precision, recall, f1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None, labels=self.labels)

        # Printing metrics for each emotion
        for idx, label in enumerate(self.labels):
            print(f"Epoch {epoch + 1} Dev Evaluation - Emotion: {label}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision[idx]}")
            print(f"Recall: {recall[idx]}")
            print(f"F1-score: {f1[idx]}")
            print("------------------------------")
# evaluation on the training set
    def evaluate_on_test_dataset(self):
        true_labels = []
        predicted_labels = []
        for _, row in self.df_test.iterrows():
            emotion_class, text = row
            actual_label = emotion_class
            true_labels.append(actual_label)

            # Making predictions using our classifier
            data_instance = EmotionSample([], text)
            max_score = -float('inf')
            predicted_label = None
            for label in self.labels:
                prediction = self.predict(data_instance.features, label)
                if prediction > max_score:
                    max_score = prediction
                    predicted_label = label
            predicted_labels.append(predicted_label)

        # Computing evaluation metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None, labels=self.labels)

        # Printing metrics for each emotion
        for idx, label in enumerate(self.labels):
            print(f"Test Evaluation - Emotion: {label}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision[idx]}")
            print(f"Recall: {recall[idx]}")
            print(f"F1-score: {f1[idx]}")
            print("------------------------------")

    # Predicting the label for a given feature list
    def predict(self, features, label_to_predict):
        score = sum(self.weights[label_to_predict].get(feature, 0.0) for feature in features)
        return 1.0 if score >= 0 else -1.0

    # Updating weights based on the prediction error
    def update_weights(self, features, label, y_true):
        for feature_label in features:
            if feature_label in self.weights[label].keys():
                self.weights[label][feature_label] += self.eta * y_true
            else:
                self.weights[label][feature_label] = self.eta * y_true