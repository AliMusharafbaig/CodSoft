
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sales_data = pd.read_csv('/content/advertising.csv')
sales_data.head()
print("Number of missing values in each column",sales_data.isnull().sum())
# For TV vs Sales
sns.scatterplot(x=sales_data['TV'], y=sales_data['Sales'])
plt.title('TV Advertising Expenditure vs Sales')
plt.show()
# For Radio vs Sales
sns.scatterplot(x=sales_data['Radio'], y=sales_data['Sales'])
plt.title('Radio Advertising Expenditure vs Sales')
plt.show()
#For Newspaper vs sales
sns.scatterplot(x=sales_data['Newspaper'],y=sales_data['Sales'])
plt.title('Newspaper Advertsiing VS SALES')
plt.show()
X=sales_data.drop(['Sales'],axis=1)
Y=sales_data['Sales']
X.head()
Y.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = LinearRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
r2_train = model.score(X_train, Y_train)
print('R-squared for training data:', r2_train)

X_test_prediction = model.predict(X_test)
r2_test = model.score(X_test, Y_test)
print('R-squared for test data:', r2_test)

mae_train = mean_absolute_error(Y_train, X_train_prediction)
print('Mean Absolute Error:', mae_train)
mae_test = mean_absolute_error(Y_test, X_test_prediction)
print('Mean Absolute Error:', mae_test)