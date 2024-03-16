import pandas as pd
from models.naive_model import NaiveModel

df = pd.read_csv('mnist_784_csv.csv')
model = NaiveModel()
model.fit(df)
model.save('model.pkl')
new_data = df.sample(n=10)
new_data.to_csv('new_data.csv', index=False)
