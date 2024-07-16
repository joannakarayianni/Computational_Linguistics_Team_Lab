from baseline_classifier.multilabelperceptron.emotion import EmotionSample
from baseline_classifier.multilabelperceptron.multilabelperceptron import MultiLabelPerceptron


def run_perceptron(data_loader):
    print("Running Multilabel Perceptron algorithm")
    
    # Labels
    labels = ['joy', 'anger', 'fear', 'sadness', 'disgust','guilt','shame']

    # MultiLabelPerceptron instance
    mlp_instance = MultiLabelPerceptron(data_loader, labels=labels, train_iterations=10, eta=0.1)

    # Training
    mlp_instance.training_of_perceptron()
    

if __name__ == "__main__":
    run_perceptron()