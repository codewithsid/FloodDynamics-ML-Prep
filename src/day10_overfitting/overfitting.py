import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("data/flood_data_harder.csv")
#df = df.dropna()

print("\nRaw data:")
print(df)

print("\nMissing values per column:")
print(df.isnull().sum())


#Object creation
simple_scaler = StandardScaler()
engineered_scaler = StandardScaler()
full_scaler = StandardScaler()

simple_model = LogisticRegression()
engineered_model = LogisticRegression()
full_model = LogisticRegression()
imputer = SimpleImputer(strategy = "mean")

#Original (Simple) Dataset
x_simple = df[["rainfall", "temperature", "soil_moisture"]]
y = df["flood"]

#Modification of DataFrame
df["rainfall_soil_interaction"] = df["rainfall"] * df["soil_moisture"]
df["temp_squared"] = df["temperature"] ** 2
df["rainfall_per_temp"] = df["rainfall"] / (df["temperature"] + 1) #using 1 to avoid dividing by zero
df["high_rainfall"] = (df["rainfall"] > 10).astype(int)
df["high_risk"] = ((df["rainfall"] > 30) & (df["soil_moisture"] > 0.5)).astype(int)
df["random_noise"] = np.random.rand(len(df))
df["rainfall_threshold"] = (df["rainfall"] > 30).astype(int)

#Engineered Model
x_engineered = df[[
    "rainfall",
    "temperature",
    "soil_moisture",
    "rainfall_soil_interaction",
    "temp_squared",
    "rainfall_per_temp"
]]

#Overloaded Model
x_full = df[[
    "rainfall",
    "temperature",
    "soil_moisture",
    "rainfall_soil_interaction",
    "temp_squared",
    "rainfall_per_temp",
    "high_rainfall",
    "high_risk",
    "random_noise"
]]

y = df["flood"]

#adding noise to explode models
for i in range(10):
    df[f"noise{i}"] = np.random.rand(len(df))

#simple dataset
simple_accuracies = [] #Takes average of accuracies for better estimate
for i in range(10): #Many splits -> many results -> average them
    x_train_simple, x_test_simple, y_train_simple, y_test_simple = train_test_split(x_simple, y, test_size = 0.4)#No random state (different splits each time)
   
    x_train_simple_imputed = imputer.fit_transform(x_train_simple)
    x_test_simple_imputed = imputer.transform(x_test_simple)
    x_train_simple_scaled = simple_scaler.fit_transform(x_train_simple_imputed)
    x_test_simple_scaled = simple_scaler.transform(x_test_simple_imputed)

    simple_model.fit(x_train_simple_scaled, y_train_simple)
    y_pred_simple = simple_model.predict(x_test_simple_scaled)
    y_prob_simple = simple_model.predict_proba(x_test_simple_scaled)

    accuracy_simple = accuracy_score(y_test_simple, y_pred_simple)
    simple_accuracies.append(accuracy_simple)
avg_simple_accuracy = np.mean(simple_accuracies)

#engineered dataset
engineered_accuracies = []
for i in range(10):
    x_train_engineered, x_test_engineered, y_train_engineered, y_test_engineered = train_test_split(x_engineered, y, test_size = 0.4)#No random state (different splits each time)
    x_train_engineered_imputed = imputer.fit_transform(x_train_engineered)
    x_test_engineered_imputed = imputer.transform(x_test_engineered)
    x_train_engineered_scaled = engineered_scaler.fit_transform(x_train_engineered_imputed)
    x_test_engineered_scaled = engineered_scaler.transform(x_test_engineered_imputed)  

    engineered_model.fit(x_train_engineered_scaled, y_train_engineered)
    y_pred_engineered = engineered_model.predict(x_test_engineered_scaled)
    y_prob_engineered = engineered_model.predict_proba(x_test_engineered_scaled)

    accuracy_engineered = accuracy_score(y_test_engineered, y_pred_engineered)
    engineered_accuracies.append(accuracy_engineered)
avg_engineered_accuracy = np.mean(engineered_accuracies)

#full dataset
full_accuracies = []
for i in range(10):
    x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(x_full, y, test_size = 0.4)#No random state (different splits each time)

    x_train_full_imputed = imputer.fit_transform(x_train_full)
    x_test_full_imputed = imputer.transform(x_test_full)
    x_train_full_scaled = full_scaler.fit_transform(x_train_full_imputed)
    x_test_full_scaled = full_scaler.transform(x_test_full_imputed)

    full_model.fit(x_train_full_scaled, y_train_full)
    y_pred_full = full_model.predict(x_test_full_scaled)
    y_prob_full = full_model.predict_proba(x_test_full_scaled)

    accuracy_full = accuracy_score(y_test_full, y_pred_full)
    full_accuracies.append(accuracy_full)
avg_full_accuracy = np.mean(full_accuracies)

print("\nSimple Model Accuracy:", avg_simple_accuracy)
print("\nEngineered Model Accuracy:", avg_engineered_accuracy)
print("\nFull Model Accuracy:", avg_full_accuracy)

print("\nModel Comparison:")
print("\nSimple Model:", simple_model.coef_)
print("\nEngineered Model:", engineered_model.coef_)
print("\nFull Model:", full_model.coef_)