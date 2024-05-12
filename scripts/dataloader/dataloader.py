import pandas as pd


class DataLoader:

    def __init__(self, path_to_train, path_to_val, path_to_test):
        # load data
        # For training and testing
        self.df_train = self.load_data_from_csv_file(path_to_train)

        self.df_val = self.load_data_from_csv_file(path_to_val)
        
        self.df_test = self.load_data_from_csv_file(path_to_test)

        self.y_train_labels = self.get_y_labels(self.df_train)

        self.y_val_labels = self.get_y_labels(self.df_val)

        self.y_test_labels = self.get_y_labels(self.df_test)

    """
        read the file and return the dataframe
    """

    def load_data_from_csv_file(self, file_name):

        df = None
        try:
            df = pd.read_csv(file_name, header=None, on_bad_lines="skip")
        except Exception as e:
            print("Error:", e)
        return self.__filter_invalid_rows(df)
    
    """
        remove invalid rows
    """

    def __filter_invalid_rows(self, df):

        valid_emotions = ["joy", "fear", "shame", "disgust", "guilt", "anger", "sadness"]
        filtered_df = df[df.iloc[:, 0].str.split(",", expand=True)[0].isin(valid_emotions)] 
        return filtered_df
    
    """
        Get all labels from a df as list 
    """

    def get_y_labels(self, df):
        return df.iloc[:, 0]
