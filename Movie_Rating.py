# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.impute import SimpleImputer
# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#Data acquisition of the movies dataset
movies_data = pd.read_csv('/content/movies.dat', sep='::', engine='python', encoding='latin1')
movies_data.columns =['MovieIDs','MovieName','Category']
movies_data.dropna(inplace=True)
print(movies_data.head())
# Checking the number of missing values in each column
print("Number of missing values in each column:\n", movies_data.isnull().sum())
#data analysis on Rating data

ratings_data = pd.read_csv('/content/ratings.dat', sep='::', engine='python', encoding='latin1')
ratings_data.columns=['ID','MovieID','Ratings','TimeStamp']
ratings_data.dropna(inplace=True)
print(ratings_data.head())
#checking missing values
print("Checking for missing values",ratings_data.isnull().sum())
#data analysis on Rating data

users_data = pd.read_csv('/content/users.dat', sep='::', engine='python', encoding='latin1')
users_data.columns=['UserID','Gender','Age','Occupation','Zip-code']
users_data.dropna(inplace=True)
print(users_data.head())
#checking missing values
print("Checking for missing values",users_data.isnull().sum())

users_data.replace({'Gender': {'M': 0, 'F': 1}}, inplace=True)
users_data.head()
df = pd.concat([movies_data, ratings_data,users_data], axis=1)
df.head()
#drooping null values
df.dropna(inplace=True)

df.head(20)
df3 = df.drop(["TimeStamp", "Zip-code", "Occupation"], axis=1)
df3.dropna(inplace=True)
df3.head(10)

df3.isnull().sum()
data_final=df3
data_final.isnull().sum()
data_final.head(30)
sns.countplot(x=data_final['Age'],hue=data_final['Ratings'])
sns.countplot(x=data_final['Gender'],hue=data_final['Ratings'])
data_final.Age.plot.hist(bins=25)
plt.ylabel("MovieID")
plt.xlabel("Ratings")
data_final['Age'].plot.hist(bins=15)
X = df3.drop(['Ratings', 'MovieName', 'MovieID', 'Category'], axis=1)
Y = df3['Ratings']

X.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
X_train.dropna(inplace=True)
Y_train = Y_train[X_train.index]
X_test.dropna(inplace=True)
Y_test = Y_test[X_test.index]
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
