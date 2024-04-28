import dataprocessor.dataloader as dp
import baseline_classifier.main_nb as nb

if __name__ == "__main__":
    path_to_gold = 'datasets/isear-val.csv'
    path_to_pred = 'datasets/isear-val-prediction.csv'

    path_to_train = 'datasets/isear-train.csv'
    path_to_test = 'datasets/isear-test.csv'

    data_loader = dp.DataLoader(path_to_train, path_to_test, path_to_gold, path_to_pred)

    # data for training
    df_train = data_loader.train_data

    # data for testing
    df_test = data_loader.test_data

    # classifier
    classifier = nb.run_naive_bayes(df_train, df_test)    
    