import pandas as pd
import pickle


class NaiveModel:
    """
    Un modelo simple que calcula la media de cada característica durante el entrenamiento y normaliza las
    características durante la predicción al dividir cada valor por la media correspondiente.
    """

    def __init__(self):
        """
        Inicializa un objeto NaiveModel.

        Attributes:
            self.means (pd.Series): Una Serie de pandas que almacena las medias de cada característica del DataFrame
            durante el entrenamiento.
        """
        self.means = None

    def fit(self, df: pd.DataFrame):
        """
        Entrena el modelo calculando la media de cada característica del DataFrame proporcionado.

        Args:
            df (pd.DataFrame): El DataFrame de pandas que contiene los datos de entrenamiento.
        """
        self.means = df.mean()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones normalizando las características del DataFrame proporcionado.

        Args:
            df (pd.DataFrame): El DataFrame de pandas que contiene los datos a predecir.

        Returns:
            pd.DataFrame: Un nuevo DataFrame con características normalizadas.
        """
        df_copy = df.copy()
        for column in df_copy.columns:
            df_copy[column] = df_copy[column] / self.means[column]
        return df_copy

    def save(self, filename: str):
        """
        Guarda las medias calculadas del modelo en un archivo utilizando pickle.

        Args:
            filename (str): El nombre del archivo donde se guardarán las medias.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.means, f)

    def load(self, filename: str):
        """
        Carga las medias previamente guardadas desde un archivo utilizando pickle.

        Args:
            filename (str): El nombre del archivo desde donde se cargarán las medias.
        """
        with open(filename, 'rb') as f:
            self.means = pickle.load(f)
