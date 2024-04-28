import pandas as pd


class DataLoader:

    def __init__(self, path_to_train, path_to_test, path_to_gold, path_to_pred):
        # load data
        # For training and testing
        self.df_train = self.__filter_invalid_rows(self.load_data_from_csv_file(path_to_train))
        
        self.df_test = self.load_data_from_csv_file(path_to_test)

    # read the file and return the dataframe
    def load_data_from_csv_file(self, file_name):

        df = None

        try:
            df = pd.read_csv(file_name, header=None, on_bad_lines="skip")
        except Exception as e:
            print("Error:", e)

        return df
    
    def __filter_invalid_rows(self, df_train):

        valid_emotions = ["joy", "fear", "shame", "disgust", "guilt", "anger", "sadness"]
        filtered_df_train = df_train[df_train.iloc[:, 0].str.split(",", expand=True)[0].isin(valid_emotions)] 
        return filtered_df_train
