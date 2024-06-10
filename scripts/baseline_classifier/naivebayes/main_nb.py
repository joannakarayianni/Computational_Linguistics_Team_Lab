from baseline_classifier.naivebayes.naive_bayes import NaiveBayes
import evaluation.evaluation_metrics as eval

def run_naive_bayes(data_loader):

    df_train = data_loader.df_train
    df_val = data_loader.df_val
    df_test = data_loader.df_test

    print("Running Naive Bayes algorithm")

    nb_instance = NaiveBayes()

    ####################### Train the classifier  ####################### 
    nb_instance.construct_dictionary_and_vocab(df_train) 
    nb_instance.train_the_classifer()

    ####################### Evaluate the classifier on the training, validation and test sets ####################### 
    print("\n**Evaluation Report on Training data:::**")
    evaluate_nb(df_train, nb_instance, output_file="scripts/baseline_classifier/naivebayes/incorrect_predictions-train.txt")
    print("\n**Evaluation Report on Validation data:::**")
    evaluate_nb(df_val, nb_instance, output_file="scripts/baseline_classifier/naivebayes/incorrect_predictions-val.txt")
    print("\n**Evaluation Report on Test data:::**")
    evaluate_nb(df_test, nb_instance, output_file="scripts/baseline_classifier/naivebayes/incorrect_predictions-test.txt")

def evaluate_nb(df_test, nb_instance, output_file):

    ####################### Test the classifier ####################### 
    
    labels = ['joy', 'anger', 'fear', 'sadness', 'disgust','guilt','shame']
    predicted_labels = []
    true_labels = []
    # List to store indices of incorrect predictions
    incorrect_indices = []

    # Initialize variables for counting correct predictions
    correct_predictions = 0
    total_samples = 0  # To count the total number of samples

    for index, row in df_test.iterrows():
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
        else:
            # Store the index and details of incorrect predictions
            incorrect_indices.append((index, actual_label, predicted_label, text))

        total_samples += 1

    ####################### Calculate accuracy of the classifier ####################### 
    accuracy = correct_predictions / total_samples
    eval.test_evaluation(true_labels, predicted_labels, labels)
    print("Accuracy is ",accuracy)

    ####################### Write details of incorrect predictions to a file ####################### 
    with open(output_file, 'w') as f:
        f.write("Index\tActual\tPredicted\tText\n")
        for idx, actual, predicted, text in incorrect_indices:
            f.write(f"{idx}\t{actual}\t{predicted}\t{text}\n")
    
    print(f"\nIncorrect predictions have been written to {output_file}.")


if __name__ == "__main__":
    run_naive_bayes()