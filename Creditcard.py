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
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
credit_data=pd.read_csv('/content/creditcard.csv')
credit_data.head(10)
print("the null or incomplete values are",credit_data.isnull().sum())
#incomplete or missing vlaues are presnet in 8 columns our of 31
credit_data.dropna(inplace=True)
credit_data.isnull().sum()
#All missing/null values have been removed now moving further
#Splitting the Target Variable from others
X=credit_data.drop(['Class'],axis=1)
Y=credit_data['Class']
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=2)
X_resampled, Y_resampled = smote.fit_resample(X_scaled, Y)

#Some data visualziation
sns.countplot(x='Class', data=credit_data)
plt.title('Class Distribution (0: Genuine, 1: Fraud)')
plt.show()
credit_data.drop(['Class'], axis=1).hist(figsize=(20,20), bins=50)
plt.suptitle('Histograms of Features')
plt.show()
sns.boxplot(x='Class', y='Amount', data=credit_data)
plt.ylim(0, 300)  # Adjust as needed to focus on the majority of data
plt.title('Transaction Amount Distribution by Class')
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=2)



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