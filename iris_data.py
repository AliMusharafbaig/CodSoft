import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
iris_data=pd.read_csv('/content/IRIS.csv')
print(iris_data.head(20))
print("The number of missing value in the data set",iris_data.isnull().sum())
x=iris_data.drop(columns=['species'])
Y=iris_data['species']
print(x.head())
print(Y.head())


#Y.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}, inplace=True)

print(Y.head())

x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=2)
model = LogisticRegression()
model.fit(x_train, Y_train)
x_train_prediction = model.predict(x_train)
print(x_train_prediction)
training_data_accuracy = accuracy_score(Y_train, x_train_prediction)
print('Accuracy score of training data:', training_data_accuracy)
x_test_prediction = model.predict(x_test)
print(x_test_prediction)
test_data_accuracy = accuracy_score(Y_test, x_test_prediction)
print('Accuracy score of test data:', test_data_accuracy)