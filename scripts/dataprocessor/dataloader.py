import pandas as pd


class DataLoader:

    def __init__(self, path_to_train, path_to_test, path_to_gold, path_to_pred):
        # load data
        # For training and testing
        self.train_data = self.load_data_from_csv_file(path_to_train)
        self.test_data = self.load_data_from_csv_file(path_to_test)

    # read the file and return the dataframe
    def load_data_from_csv_file(self, file_name):

        df = None

        try:
            df = pd.read_csv(file_name, on_bad_lines="skip")
        except Exception as e:
            print("Error:", e)

        return df
