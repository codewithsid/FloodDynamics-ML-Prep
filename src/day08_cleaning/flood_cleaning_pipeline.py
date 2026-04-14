import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/flood_data.csv")
#df = df.dropna()

print("\nRaw data:")
print(df)

print("\nMissing values per column:")
print(df.isnull().sum())

x = df[["rainfall", "temperature", "soil_moisture"]]
y = df["flood"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)

#Handling Missing Values
imputer = SimpleImputer(strategy = "mean") 

#Learn from training data
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

print("\nImputed training data:")
print(x_train_imputed)

#Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_imputed)
x_test_scaled = scaler.transform(x_test_imputed)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
y_prob = model.predict_proba(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy after cleaning:", accuracy)
