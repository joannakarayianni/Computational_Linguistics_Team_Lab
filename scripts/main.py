import dataprocessor.dataloader as dp
import baseline_classifier.main_nb as nb
import baseline_classifier.multilabelperceptron.main_mlp as mlp
import advanced_classifier.word_embeddings.retrain_word2vec as customEmbeddings

if __name__ == "__main__":
    path_to_gold = 'datasets/isear-val.csv'
    path_to_pred = 'datasets/isear-val-prediction.csv'

    path_to_train = 'datasets/isear-train.csv'
    path_to_test = 'datasets/isear-test.csv'

    data_loader = dp.DataLoader(path_to_train, path_to_test, path_to_gold, path_to_pred)

    # data for training
    df_train = data_loader.df_train

    # data for testing
    df_test = data_loader.df_test

    # naive bayes classifier
    nb.run_naive_bayes(df_train, df_test)   

    # run perceptron classifier 
    mlp.run_perceptron(df_train, df_test)

    custom_embeddings = customEmbeddings.WordEmbeddingTrainer(df_train)

    X_train_embeddings, y_train_labels = custom_embeddings.get_embeddings_matrix()

    print(X_train_embeddings, y_train_labels)
