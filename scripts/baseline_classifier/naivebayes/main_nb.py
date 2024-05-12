#from naive_bayes import NaiveBayes
from baseline_classifier.naivebayes.naive_bayes import NaiveBayes
import evaluation.evaluation_metrics as eval

def run_naive_bayes(df_train, df_test):
    print("Running Naive Bayes algorithm")

    nb_instance = NaiveBayes()

    ####################### Train the classifier  ####################### 
    nb_instance.construct_dictionary_and_vocab(df_train) 
    nb_instance.train_the_classifer()

   
    evaluate_nb(df_test, nb_instance)

def evaluate_nb(df_test, nb_instance):

     ####################### Test the classifier ####################### 
    
    labels = ['joy', 'anger', 'fear', 'sadness', 'disgust','guilt','shame']
    predicted_labels = []
    true_labels = []

    # Initialize variables for counting correct predictions
    correct_predictions = 0
    total_samples = 0  # To count the total number of samples

    for _, row in df_test.iterrows():
        # unpacking the row fields.
        emotion_class, text = row
                    
        actual_label = emotion_class
        true_labels.append(actual_label)

        # Make a prediction using your classifier
        predicted_label = nb_instance.get_the_best_class(text)
        predicted_labels.append(predicted_label)

        # Check if the prediction is correct
        if predicted_label == actual_label:
            correct_predictions += 1

        total_samples += 1

    ####################### Calculate accuracy of the classifier ####################### 
    accuracy = correct_predictions / total_samples
    print("\nEvaluation Report :::")
    eval.test_evaluation(true_labels, predicted_labels, labels)
    print("Accuracy is ",accuracy)

if __name__ == "__main__":
    run_naive_bayes()