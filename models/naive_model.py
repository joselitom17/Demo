import pandas as pd
import pickle


class NaiveModel:

    def __init__(self):
        self.means = None

    def fit(self, df: pd.DataFrame):
        self.means = df.mean()

    def predict(self, df: pd.DataFrame):
        df_copy = df.copy()
        for column in df_copy.columns:
            df_copy[column] = df_copy[column] / self.means[column]
        return df_copy

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.means, f)

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            self.means = pickle.load(f)
