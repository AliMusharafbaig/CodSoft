# CodSoft
*Task 1*
This project focuses on building a machine learning model to predict whether a passenger on the Titanic survived or not. It utilizes the classic Titanic dataset, which contains information about individual passengers, such as age, gender, ticket class, fare, and more
Key Steps:

Data Loading: The Titanic dataset is loaded from a CSV file.
Data Exploration: Initial data exploration is performed, including checking dimensions, missing values, and summary information.
Data Preprocessing: Missing values are handled by replacing them with mean values, and categorical columns (Sex, Embarked) are converted into numerical form.
Feature Selection: Irrelevant columns (PassengerId, Name, Ticket) are dropped, leaving relevant features for prediction.
Data Splitting: The dataset is split into training and testing sets using the train_test_split function.
Model Training: A logistic regression model is initialized and trained using the training data.
Prediction: The trained model is used to predict survival outcomes on both the training and testing sets.
Accuracy Calculation: The accuracy of the model is calculated based on the predicted results.
Results Display: The training and testing accuracy scores are displayed to assess the model's performance.


**Very Important steps below as code was created on Google Colab**

Usage Instructions:

Clone or download the repository.
Ensure the 'tested.csv' dataset file is in the same directory as the script.
Set up the necessary Python environment with required libraries.
Run the script to load, preprocess, train the model, and calculate accuracy.
View the accuracy scores on the console to evaluate model performance.
