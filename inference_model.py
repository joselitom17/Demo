import pandas as pd
from models.naive_model import NaiveModel  # Importa la clase NaiveModel del archivo naive_model.py

# Inicializa un objeto NaiveModel
model = NaiveModel()

# Carga un modelo pre-entrenado desde un archivo llamado 'model.pkl'
model.load('model.pkl')

# Lee los nuevos datos desde un archivo CSV llamado 'new_data.csv' utilizando pandas
new_data = pd.read_csv('new_data.csv')

# Realiza predicciones en los nuevos datos utilizando el modelo cargado
predictions = model.predict(new_data)

# Rellena los valores nulos en las predicciones con las medias de las características de los nuevos datos
predictions.fillna(new_data.mean(), inplace=True)

# Guarda las predicciones en un archivo CSV llamado 'predictions.csv', sin incluir los índices de fila
predictions.to_csv('predictions.csv', index=False)
