# Proyecto de Machine Learning con Python

Este proyecto utiliza Python para crear un modelo de machine learning muy simple llamado `NaiveModel`. Este modelo calcula la media de cada característica en un conjunto de datos de entrenamiento y luego utiliza estas medias para transformar nuevos datos.

## Estructura del proyecto

El proyecto consta de tres scripts de Python:

1. `naive_model.py`: Define la clase `NaiveModel`, que incluye métodos para entrenar el modelo, hacer predicciones con el modelo, y guardar y cargar el modelo.

2. `train_model.py`: Utiliza la clase `NaiveModel` para entrenar un modelo con un conjunto de datos y luego guarda el modelo entrenado en el disco.

3. `inference_model.py`: Carga un modelo previamente entrenado desde el disco y lo utiliza para hacer predicciones sobre nuevos datos.

## Cómo usar este proyecto

Para utilizar este proyecto, sigue estos pasos:

1. Ejecuta el script `train_model.py` para entrenar el modelo y guardarlo en el disco.

2. Ejecuta el script `inference_model.py` para cargar el modelo entrenado y hacer predicciones sobre nuevos datos.

Por favor, asegúrate de tener instaladas las bibliotecas de Python necesarias, que incluyen pandas y pickle.

## Conjunto de datos

Este proyecto utiliza el conjunto de datos MNIST, que es un conjunto de imágenes de dígitos escritos a mano. Cada imagen es de 28x28 píxeles, y cada píxel es una característica. Por lo tanto, cada imagen se representa como una fila de 784 características (píxeles) en el conjunto de datos.
