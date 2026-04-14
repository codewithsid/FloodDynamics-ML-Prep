import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data/flood_data.csv")

print("First 5 Rows of the Dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values per column:")
print(df.isnull().sum())

x = df[["rainfall", "temperature", "soil_moisture"]]
y = df["flood"]

print("\nFeature matrix shape:", x.shape)
print("Label vector shape:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print("\nTraining feature shape:", x_train.shape)
print("Testing feature shape:", x_test.shape)

scaler = StandardScaler() #creating the scalar object
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)    

model = LogisticRegression() #creating the model object
model.fit(x_train_scaled, y_train) 
y_pred = model.predict(x_test_scaled)
y_prob = model.predict_proba(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred)) 

#comparting actual vs. predicted
results = x_test.copy() #copying the test features to a new dataframe for comparison
results["actual_flood"] = y_test.values #adding the actual flood values to the results dataframe
results["predicted_flood"] = y_pred #adding the predicted flood values to the results dataframe
results["predicted_probability"] = y_prob[:, 1] #adding the predicted probabilities of flood occurrence to the results dataframe (only the probability of the positive class, which is why we use [:, 1])

print("\nTest set results")
print(results) 

#Predicting for new data
new_data = pd.DataFrame([{
    "rainfall": 12,
    "temperature": 75,
    "soil_moisture": 0.20
}])
new_data_scaled = scaler.transform(new_data)
new_pred = model.predict(new_data_scaled)[0]#predicting the flood occurrence for the new data and accessing the predicted probability of flood occurrence for the new data (only the probability of the positive class, which is why we use [0][1])
new_prob = model.predict_proba(new_data_scaled)[0][1]

print("\nNew Sample")
print(new_data)

print("\nPredicted Probability of Flood")
print(new_prob)

print("\nPredicted class:")
print(new_pred)

print("\nModel Coefficients:")
print(model.coef_)