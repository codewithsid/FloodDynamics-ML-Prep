import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = {
    "rainfall": [5, 15, 25, 35, 45, 60, 70, 80],
    "flood": [0, 0, 0, 1, 1, 1, 1, 1]

}

df = pd.DataFrame(data)

x = df[["rainfall"]]
y = df["flood"]

print(df)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = LogisticRegression() #creating model object
model.fit(x_train, y_train) #fitting the model, finds decision boundary

y_prob = model.predict_proba(x_test) #predicting probabilities for the test data
y_pred = model.predict(x_test) #predicting flood occurence from test data of rainfall
#this converts probabilities to binary predictions based on a threshold (default is 0.5)

#evaluation
accuracy = accuracy_score(y_test, y_pred) #calculating accuracy by comparing predicted values with actual values
print("Accuracy:", accuracy) #printing the accuracy of the model

#output
print("\nx_test:\n", x_test) #printing the test data for independent variable
print("\ny_test (actual):\n", y_test) #printing the actual values of the dependent variable for the test data
print("\ny_prob (predicted probabilities):\n", y_prob) #printing the predicted probabilities of the dependent variable for the test data
print("\ny_pred (predicted classes):\n", y_pred) #printing the predicted classes of the dependent variable for the test data    

#mini experiments
new_rain = [[50]]
print("flood probability for rainfall of 50:", model.predict_proba(new_rain)[0][1]) #predicting the probability of flood occurrence for a new rainfall value of 50
print("flood prediction for rainfall of 50:", model.predict(new_rain)[0]) #predicting the flood occurrence for a new rainfall value of 50 (binary)

prob = model.predict_proba(new_rain)[0][1] #storing the predicted probability of flood occurrence for a new rainfall value of 50
if prob > 0.7:
    print("Flood Likely")
else:
    print("Flood Unlikely")

#graphical representation of the decision boundary
plt.scatter(x, y, color='blue', label='data points') #scatter plot for the original data points
plt.xlabel("Rainfall")
plt.ylabel("Flood")
plt.title("Logistic Regression: Decision Boundary")
#plotting the decision boundary
plt.plot(x, model.predict_proba(x)[:, 1], color='orange', label='predicted probabilities') #plotting the predicted probabilities as a curve
plt.legend() #adding legend to the plot
plt.savefig("outputs/day04_classification.png") #saving the plot as an image in outputs folder
plt.show() #displaying the plot
