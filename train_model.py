import pandas as pd
from models.naive_model import NaiveModel

df = pd.read_csv('mnist_784_csv.csv')
model = NaiveModel()
model.fit(df)
new_data = df.sample(n=10)
new_data.to_csv('new_data.csv', index=False)
model.save('model.pkl')



