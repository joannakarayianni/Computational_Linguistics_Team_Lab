import dataloader.dataloader as dp
import baseline_classifier.naivebayes.main_nb as nb
import baseline_classifier.multilabelperceptron.main_mlp as mlp
import advanced_classifier.sequential_nn.sequential_nn_word2vec as seq_nn1
import advanced_classifier.sequential_nn.sequential_nn_word2vec_dropout as seq_nn2
import advanced_classifier.sequential_nn.sequential_nn_fine_tuned_word2vec as seq_nn3
import advanced_classifier.sequential_nn.sequential_nn_with_fine_tuned_word2vec_tfidf as seq_nn4
import advanced_classifier.sequential_nn.sequential_nn_glove as seq_nn5
import advanced_classifier.lstm.lstm_glove_simple as lstmglove
import advanced_classifier.lstm.lstm_tfidf_simple as lstmtfidf
import advanced_classifier.lstm.bi_lstm as bilstm
import advanced_classifier.lstm.lstm as lstm


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

    # Phase 2 code for advanced classifier.

    print("Running Sequential NN with word2vec!!!")

    seq_nn1_instance = seq_nn1.SequentialNNWord2Vec(data_loader)
    
    seq_nn1_instance.train()

    print("Running Sequential NN with word2vec & dropout!!!")

    seq_nn2_instance = seq_nn2.SequentialNNWord2VecDropout(data_loader)
    
    seq_nn2_instance.train()

    print("Running Sequential NN with fine-tuned word2vec!!!")

    seq_nn3_instance = seq_nn3.SequentialNNWithFineTunedW2Vec(data_loader)

    seq_nn3_instance.train()

    print("Running Sequential NN with fine-tuned word2vec & TF-IDF!!!")

    seq_nn4_instance = seq_nn4.SequentialNNCustomWord2VecTFIDF(data_loader)

    seq_nn4_instance.train()

    print("Sequential NN with glove (6B, 300d)!!!")

    seq_nn5_instance = seq_nn5.SequentialNNGlove(data_loader, 'scripts/advanced_classifier/word_embeddings/glove.6B/glove.6B.300d.txt')
    
    seq_nn5_instance.train()


    print(" ******************** Results for LSTM with word Embeddings ********************")
    nn3_instance = lstm.LongShortTerm(data_loader)
    nn3_instance.train()
    
    print(" ******************** Results for LSTM with TF-IDF Embeddings ********************")
    nn4_instance = lstmtfidf.LSTM_Tf_Idf(path_to_train, path_to_val, path_to_test)
    nn4_instance.build_model()
    nn4_instance.train()
    nn4_instance.evaluate_test_set()
    
    print(" ******************** Results for Bi-LSTM with word Embeddings ********************")
    nn5_instance = bilstm.BiLongShortTerm(data_loader)
    nn5_instance.train() 
    
    print(" ******************** Results for LSTM with GloVE Embeddings ********************")
    nn6_instance = lstmglove.LSTM_GloVe(data_loader)
    nn6_instance.train()

