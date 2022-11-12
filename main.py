import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


import matplotlib.pyplot as plt


data = pd.read_csv("Dataset/winequality-white.csv",index_col=False).reset_index()
print(data.shape)

corr = data.corr()

print(corr["quality"].sort_values(ascending=False))

data.hist()
plt.show()

x_train,x_test =train_test_split(data,test_size=0.1,shuffle=True)



print(x_train.shape)
print(x_test.shape)

n_test = x_test.drop("quality",axis=1)
n_test_label = x_test["quality"].copy()

n_train = x_train.drop("quality",axis=1)
n_train_label = x_train["quality"].copy()

# print(n_train.head())
# print("------------------------------")
# print(n_train_label.head())


model = RandomForestRegressor(random_state=42)
model.fit(n_train,n_train_label)



result = model.score(n_test,n_test_label)
print(f"Accuracy of this Model - {result}")
