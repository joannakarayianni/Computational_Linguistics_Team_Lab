"""Creating baseline classifier - MultiLabelPerceptron 
(for multilabel classification task - Emotion Recognition on text) 
Parameters: Training instances/ training data, learning rate- eta, labels (emotion labels-classes), iterations on training set
Attributes: weights, features"""


from baseline_classifier.multilabelperceptron.emotion import EmotionSample
import evaluation.evaluation_metrics as eval
from sklearn.model_selection import KFold
import pandas as pd

class MultiLabelPerceptron:
    def __init__(self, dataloader, labels, train_iterations=100, eta=0.1, k_folds=5):
        self.df_train = dataloader.df_train
        self.df_val = dataloader.df_val
        self.df_test = dataloader.df_test
        self.train_iterations = train_iterations
        self.labels = labels
        self.eta = eta
        self.k_folds = k_folds
        
        # Attributes
        self.weights = {}  # Storing weights for each feature of each label
        self.initialize_weights()  # Calling function for weight initialization
        self.features = self.weights.keys()  # Set of all features with weights considered during training

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
        # Combine the training and validation datasets
        df_combined = pd.concat([self.df_train, self.df_val])
        
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_number = 0
        for train_index, val_index in kf.split(df_combined):
            df_train_fold = df_combined.iloc[train_index]
            df_val_fold = df_combined.iloc[val_index]
            fold_number+=1
            for epoch in range(self.train_iterations):
                train_correct = 0
                train_total = 0
                for _, row in df_train_fold.iterrows():
                    emotions = row[0].split()
                    text = row[1]
                    train_instance = EmotionSample(emotions, text)
                    features = train_instance.features 
                    for label in self.labels:
                        # Correct label if it matches the current label
                        y_true = 1.0 if label in train_instance.emotions else -1.0
                        # Will be updated based on features and weights during prediction
                        y_predicted = self.predict(features, label) # We call predict function
                        if y_true == y_predicted:
                            train_correct += 1
                        train_total += 1
                        if y_true != y_predicted:
                            
                            self.update_weights(features, label, y_true) # We call update_weights function

                train_accuracy = train_correct / train_total

                ####################### Run the classifier on the dev dataset ####################### 
                dev_accuracy = self.evaluate_on_dev_and_test(df_val_fold)
            
                print(f'Epoch {epoch + 1} K-fold-val {fold_number}: Training Accuracy: {train_accuracy}, Validation Accuracy: {dev_accuracy}')
            
            ####################### Run the classifier on the test dataset ####################### 
            test_accuracy = self.evaluate_on_dev_and_test(self.df_test)
            print(f'Test accuracy: {test_accuracy}')

    def evaluate_on_dev_and_test(self, df):

        labels = self.labels
        predicted_labels = []
        true_labels = []

        correct_predictions = 0
        total_samples = 0  # To count the total number of samples

        for _, row in df.iterrows():
            # unpacking the row fields.
            emotion_class, text = row
                        
            actual_label = emotion_class
            true_labels.append(actual_label)

            # Make a prediction using your classifier
            data_instance = EmotionSample([], text)
            max_score = -float('inf')
            predicted_label = None
            for label in labels:
                prediction = self.predict(data_instance.features, label)
                if prediction > max_score:
                    max_score = prediction
                    predicted_label = label

            # Check if the prediction is correct
            if predicted_label == actual_label:
                correct_predictions += 1
            total_samples += 1
            predicted_labels.append(predicted_label)
        
        accuracy = correct_predictions / total_samples
        eval.test_evaluation(true_labels, predicted_labels, labels)

        return accuracy

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
