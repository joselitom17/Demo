import pandas as pd
from models.naive_model import NaiveModel

model = NaiveModel()
model.load('model.pkl')
new_data = pd.read_csv('new_data.csv')
predictions = model.predict(new_data)
predictions.to_csv('predictions.csv')
