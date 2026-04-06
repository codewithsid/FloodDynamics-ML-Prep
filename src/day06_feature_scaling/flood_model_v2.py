import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import numpy as np

data = {
    "rainfall": [5, 15, 25, 35, 45, 60, 70, 80],
    "temperature": [5, 15, 25, 35, 45, 60, 70, 80],
    "soil_moisture": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
    "flood": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

x = df[["rainfall", "temperature", "soil_moisture"]] #multidimentsional features
y = df["flood"] #single dimensional output
#shape of model (8,3) -> 8 rows and 3 columns (features)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

scaler = StandardScaler() #creating the scaler object
x_train_scaled  = scaler.fit_transform(x_train) #fitting scaler to trainging data and transforming it
x_test_scaled = scaler.transform(x_test) #must use the same scalar as the traning data to stay consistent
new_data = [[10, 95, 0.9]] #extreme data 
new_data_scaled = scaler.transform(new_data)

model = LogisticRegression() #creating model object
model.fit(x_train_scaled, y_train) #training the model with scaled data | y_train is not scaled because it is the output variable and does not require scaling

y_pred = model.predict(x_test_scaled) #predicting the ouput from the scaled test set
y_prob = model.predict_proba(x_test_scaled) #predicting the probabilities of outputs from the scaled test set
accuracy = accuracy_score(y_test, y_pred) #calculating the accuracty from the model predictions using y_test set
print("Accuracy:", accuracy)
print("Scaled Model Coefficients:", model.coef_) #representation of the automatically assigned weights to features 

#comparing outputs
print("Unscaled prediction:", model.predict_proba(x_test))
print ("Scaled prediction:", model.predict_proba(x_test_scaled))




