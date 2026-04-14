import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("data/flood_data.csv")
#df = df.dropna()

print("\nRaw data:")
print(df)

print("\nMissing values per column:")
print(df.isnull().sum())


#Object creation
scaler = StandardScaler()
old_model = LogisticRegression()
new_model = LogisticRegression()

#Original Dataset
x = df[["rainfall", "temperature", "soil_moisture"]]
y = df["flood"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
old_model.fit(x_train_scaled, y_train)
y_pred = old_model.predict(x_test_scaled)
y_prob = old_model.predict_proba(x_test_scaled)

#Baseline model 
baseline_accuracy = accuracy_score(y_test, y_pred)
print("\nBaseline accuracy:", baseline_accuracy)

#Modification of DataFrame
df["rainfall_soil_interaction"] = df["rainfall"] * df["soil_moisture"]
df["temp_squared"] = df["temperature"] ** 2
df["rainfall_per_temp"] = df["rainfall"] / (df["temperature"] + 1) #using 1 to avoid dividing by zero
df["high_rainfall"] = (df["rainfall"] > 10).astype(int)
df["high_risk"] = ((df["rainfall"] > 30) & (df["soil_moisture"] > 0.5)).astype(int)
df["random_noise"] = np.random.rand(len(df))
df["rainfall_threshold"] = (df["rainfall"] > 30).astype(int)

#Retraining model
x_new = df [[
    "rainfall",
    "temperature", 
    "soil_moisture",
    "rainfall_soil_interaction",
    "temp_squared",
    "rainfall_per_temp", 
    "high_rainfall",
    "high_risk",
    "random_noise",
    "rainfall_threshold"
]]
y = df["flood"]

x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_new, y, test_size = 0.25, random_state=42)

#New Feature Dataset
print(df)

#Handling Missing Values
imputer = SimpleImputer(strategy = "mean") 
x_train_imputed = imputer.fit_transform(x_train_new)
x_test_imputed = imputer.transform(x_test_new)
x_train_scaled_new = scaler.fit_transform(x_train_imputed)
x_test_scaled_new = scaler.transform(x_test_imputed)

print("\nImputed training data:")
print(x_train_imputed)

new_model.fit(x_train_scaled_new, y_train_new)

y_pred = new_model.predict(x_test_scaled_new)
y_prob = new_model.predict_proba(x_test_scaled_new)

new_accuracy = accuracy_score(y_test_new, y_pred)
print("\nAccuracy after feature engineering:", new_accuracy)
print("\n------------------")

print("Baseline Accuracy:", baseline_accuracy)
print("New Accuracy:", new_accuracy)

print("\nModel Coefficients:", new_model.coef_)