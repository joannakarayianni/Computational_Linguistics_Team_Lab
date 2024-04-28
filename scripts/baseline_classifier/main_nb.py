#from naive_bayes import NaiveBayes
from baseline_classifier.naive_bayes import NaiveBayes

def run_naive_bayes(df_train, df_test):
    print("Running Naive Bayes algorithm")
    print(len(df_train), len(df_test))

    object = NaiveBayes()

    ####################### Train the classifier  ####################### 
    object.construct_dictionary_and_vocab(df_train) 
    prior_probs, likelihood_probs = object.train_the_classifer()

    ####################### Test the classifier ####################### 
    # Initialize variables for counting correct predictions
    correct_predictions = 0
    total_samples = 0  # To count the total number of samples

    for row in df_test.iterrows():
        # unpacking the row fields.
        emotion_class, text = row
                    
        actual_label = emotion_class

        # Make a prediction using your classifier
        predicted_label = object.get_the_best_class(text, prior_probs, likelihood_probs)

        # Check if the prediction is correct
        if predicted_label == actual_label:
            correct_predictions += 1

        total_samples += 1

    ####################### Calculate accuracy of the classifier ####################### 
    accuracy = correct_predictions / total_samples
    print("\nEvaluation Report :::")
    print("Accuracy is ",accuracy)
    print(f"f-score = {f_score}\nprecision = {precision}\nrecall = {recall}\nmacro-f = {macro}\nmicro-f = {micro}")

if __name__ == "__main__":
    run_naive_bayes()