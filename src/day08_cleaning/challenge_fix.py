import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("data/flood_data_broken.csv")
print("\nRaw data:")
print(df)

print("\nInitial Data Types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

#Fixing Label Column
df["flood"] = df["flood"].replace({"yes": 1}) #Changes the "yes" values to 1 in flood column
print("\nFlood Column after label fix:")
print(df["flood"])

#Safely convert values to numbers, and if conversion fails replace with NaN
numeric_columns = ["rainfall", "temperature", "soil_moisture"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors = "coerce")

df["flood"] = pd.to_numeric(df["flood"], errors = "coerce")

print("\nDataset after numeric conversion:")
print(df.dtypes)

print("\nDataset after numeric conversion:")
print(df) 

print("\nMissing values after conversion:")
print(df.isnull().sum())

#Dropping rows with missing  y labels (otherwise useless)
df = df.dropna(subset = ["flood"])
print("\nDataset after dropping rows with missing labels:")
print(df)

#Regular feature stuff
x = df[["rainfall", "temperature", "soil_moisture"]]
y = df["flood"]

print("\nFeature matrix:")
print(x)

print("\nLabel vector:")
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)

imputer = SimpleImputer(strategy = "mean")

x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

print("\nImputed training data:")
print(x_train_imputed)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_imputed)
x_test_scaled = scaler.transform(x_test_imputed)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
y_prob = model.predict_proba(x_test_scaled)

print("\nAccuracy score:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Showing Readable Results
results = x_test.copy()
results["actual_flood"] = y_test.values
results["predicted_flood"] = y_pred
results["predicted_probability"] = y_prob[:, 1]

print("\nTest results:")
print(results)
