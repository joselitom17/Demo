# Modulos y Librerias
Este es mi primer proyecto, a continuación, adjunto el enunciado del mismo.
El objetivo de este ejercicio será poner en práctica los conocimientos básicos adquiridos del lenguaje de programación Python, orientándonos sobre todo a futuros desarrollos relacionados con modelos de machine learning.

Se deberá desarrollar lo siguiente:

Un script de Python (al que llamaremos **naive_model.py**) que contenga una clase llamada **NaiveModel**. Esta clase tendrá 5 métodos públicos:

El **__init__** al que se le deberán pasar los parámetros que se consideren oportunos (puede que no haga falta pasarle ninguno).
* Un método llamado **fit** que recibirá unos datos de entrada en formato "Data Frame" de Pandas (investigar que es esto) y guardará dentro de la clase (en el self) un diccionario con la media de cada columna.
* Un método llamado **predict** que utilizará esta información guardada en la clase para transformar nuevos datos que le puedan llegar en formato DataFrame. En concreto este método deberá de retornar una versión modificada de este DataFrame en el que cada dato haya sido dividido por la media correspondiente a su columna. Se debe hacer con cuidado para no modificar los datos originales por lo que se recomienda hacer una copia profunda de los datos de entrada en una variable local del método.
* Un método llamado **save** que permita guardar en disco serializado en pickle las medias anteriormente calculadas.
* Un método llamado **load** que permita volver a cargar de disco el estado de la propia clase para poder volver a llamar a predict en una futura sesión sin tener que “reentrenar” el modelo.
* Este script (naive_model.py) se debe de meter dentro de un paquete al que llamaremos models.

Fuera de este paquete debe de haber 2 archivos más.
* Un archivo llamado **train_model.py** que importará dicho paquete, creará objeto de tipo NaiveModel, lo entrenará (calculará las medias) invocando el método "fit" con un conjunto de datos (el CSV que se especifica más abajo, y que lerermos con el métodos de read_csv de pandas) y luego serializará a disco lo aprendido mediante el método "save".
* Un archivo llamado **inference_model.py** que creará un objeto de tipo NaiveModel, leerá mediante el método "load" lo ya aprendido y realizará la inferencia de nuevos datos, la cual dejará en un fichero csv.
 