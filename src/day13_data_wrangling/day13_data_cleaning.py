import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/flood_data_brokener.csv")
print("\nMissing Data:\n", df.isnull().sum()) #summing up the null values in each column
print("\nData Description:\n", df.describe()) #describing the data to find outliers

x = df[["rainfall", "temperature", "soil_moisture"]]
y = df["flood"]

log_accuracy = []

for i in range(20):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

    #imputing missing values from the data
    imputer = SimpleImputer(strategy = "mean")
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)

    #logistic Regression
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_imputed)
    x_test_scaled = scaler.transform(x_test_imputed)

    log_model = LogisticRegression()
    log_model.fit(x_train_scaled, y_train)

    #accuracies and predictions
    y_pred_log = log_model.predict(x_test_scaled)
    y_prob_log = log_model.predict_proba(x_test_scaled)[:, 1]
    log_acc = accuracy_score(y_test, y_pred_log)
    log_accuracy.append(log_acc)

print("Average Logistic Regression Accuracy:", np.mean(log_accuracy))