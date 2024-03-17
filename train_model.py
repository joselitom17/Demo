import pandas as pd
from models.naive_model import NaiveModel  # Importa la clase NaiveModel del archivo naive_model.py

# Lee los datos MNIST desde un archivo CSV llamado 'mnist_784_csv.csv' utilizando pandas
df = pd.read_csv('mnist_784_csv.csv')

# Inicializa un objeto NaiveModel
model = NaiveModel()

# Entrena el modelo NaiveModel con los datos leídos
model.fit(df)

# Guarda el modelo entrenado en un archivo llamado 'model.pkl'
model.save('model.pkl')

# Muestra aleatoria de 10 filas de los datos originales y los guarda en un archivo CSV llamado 'new_data.csv'
new_data = df.sample(n=10)
new_data.to_csv('new_data.csv', index=False)
