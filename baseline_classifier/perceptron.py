""" Creating baseline classifier - Perceptron class 
(for binary classification task) to help us with the multilabel perceptron 
Parameters: Training instances, learning rate- eta, labels (emotion labels-classes), iterations on training set
Attributes: weights, features """

class Perceptron:
    def __init__(self, train_instances, label, train_iterations=10, eta=0.1) -> None:
        self.train_instances = train_instances
        self.train_iterations = train_iterations
        self.label = label
        self.eta = eta
        
        # Attributes
        self.weights = {} # Storing weights for each feature
        self.initialize_weights() # Calling function for weight initialization

        self.features = self.weights.keys() # Set of all features with weights considered during training


# Weight initializing function
    def initialize_weights(self):
            for train_instance in self.train_instances:
                feature_labels = train_instance.features
                for feature_label in feature_labels:
                 if feature_label not in self.weights.keys():
                    self.weights[feature_label] = 0.0


# Perceptron training with provided instances
def training_of_perceptron(self):
    # Training done for each class
    for epoch in range(self.train_iterations):
        for train_instance in self.train_instances:
            features = train_instance.features 
            # Correct label if it matches the correct emotion
            y_true = 1.0 if train_instance == self.label else -1.0
            # Will be updated based on features and weights during prediction
            y_predicted = self.predict(features)
            if y_true != y_predicted:
                self.update_weights(features, y_true)


# Label prediction for feature set returns binary prediction
def predict(self,features):
   score = sum(self.weights.get(feature, 0.0) for feature in features)
   return 1.0 if score >= 0 else -1.0


# Update weights based on features and true label
def update_weights(self,features,y_true):
  for feature_label in features:
# If feature is in weights dictionary it updates the weight by adding eta * true label
# If not initializes weight with eta * true label
     if feature_label in self.weights.keys():
        self.weights[feature_label] += self.eta * y_true
     else:
       self.weights[feature_label] = self.eta * y_true


