
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data from the csv file
titanic_data = pd.read_csv('/content/tested.csv')

# Getting insights into the data
print(titanic_data.info())

# Checking the number of missing values in each column
print("Number of missing values in each column:\n", titanic_data.isnull().sum())

# Drop the Cabin column from the data frame
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# Replace missing values in the Age column with the mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# Replace missing values in the Fare column with the mean value
titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)

# Converting categorical columns
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
print(X_train_prediction)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
print(X_test_prediction)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data:', test_data_accuracy)
