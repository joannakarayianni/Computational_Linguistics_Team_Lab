import dataloader.dataloader as dp
import baseline_classifier.naivebayes.main_nb as nb
import baseline_classifier.multilabelperceptron.main_mlp as mlp
import advanced_classifier.neural_nets.sequential_nn as nn1
import advanced_classifier.neural_nets.sequential_nn_with_tfidf as nn2

if __name__ == "__main__":
    
    path_to_train = 'datasets/isear-train.csv'
    path_to_val = 'datasets/isear-val.csv'
    path_to_test = 'datasets/isear-test.csv'

    data_loader = dp.DataLoader(path_to_train, path_to_val, path_to_test)

    # data for training
    df_train = data_loader.df_train

    # data for validation
    df_val = data_loader.df_val
    
    # data for testing
    df_test = data_loader.df_test

    # naive bayes classifier
    nb.run_naive_bayes(data_loader)   

    # run perceptron classifier 
    mlp.run_perceptron(data_loader)

    # Phase 2 code for advanced classifier. Hence, commented out.

    # nn1_instance = nn1.SequentialNN(data_loader)
    
    # nn1_instance.train()

    # nn2_instance = nn2.SequentialNNWithTFIDF(data_loader)

    # nn2_instance.train()
