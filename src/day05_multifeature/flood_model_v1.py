import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
model = LogisticRegression() #creating the model object
model.fit(x_train, y_train) #training the model

y_pred = model.predict(x_test) #predicting the ouput from the test set
y_prob = model.predict_proba(x_test) #predicting the probabilities of outputs from the test set
accuracy = accuracy_score(y_test, y_pred) #calculating the accuracty from the model predictions using y_test set
print("Accuracy:", accuracy)
print("Model Coefficients:", model.coef_) #representation of the automatically assigned weights to features 

new_data = [[10, 95, 9.5]] #indicates extrapolation -- good for bench testing
model.predict(new_data)
print("Flood prediction for new data (10, 95, 9.5):", model.predict(new_data)[0]) #predicting the flood occurrence for a new set of features (rainfall=10, temperature=95, soil_moisture=9.5)
print("Flood probability for new data (10, 95, 9.5):", model.predict_proba(new_data)[0][1]) #predicting the probability of flood occurrence for a new set of features (rainfall=10, temperature=95, soil_moisture=9.5)

#Visualization of only 2 features (rainfall and soil-moisture)
x_vis = df[["rainfall", "soil_moisture"]]
y = df["flood"] 

model_vis = LogisticRegression()
model_vis.fit(x_vis, y) #fitting truncated model

x_min = x_vis["rainfall"].min() - 5
x_max = x_vis["rainfall"].max() + 5
y_min = x_vis["soil_moisture"].min() - 0.1
y_max = x_vis["soil_moisture"].max() + 0.1

xx, yy, = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

grid = np.c_[xx.ravel(), yy.ravel()]
z = model_vis.predict(grid)
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, alpha=0.1)
plt.scatter(x_vis["rainfall"], x_vis["soil_moisture"], c=y) #c = y is color is yellow for 1 and blue for 0
plt.xlabel("Rainfall")
plt.ylabel("Soil Moisture")
plt.title("Flood Classification Boundary")
plt.savefig("outputs/day05_contourplot.png")
plt.show()
