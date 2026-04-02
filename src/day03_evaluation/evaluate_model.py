import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt

data = {
    "rainfall" : [5, 15, 25, 35, 45, 60, 70, 80, 90, 100, 110, 4, 12, 22, 32, 42, 55, 65, 75, 85, 95, 105],
    "flood" : [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

x = df[["rainfall"]] #2D array for independent variable
y = df["flood"] #1D array for dependent variable

#Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = LinearRegression() #creating linear regression model
model.fit(x_train, y_train) #using the training data split for training the model

#Making predictions on the testing data
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred) #comparning the predicted values with the actual values using Mean Squared Error
print("Mean Squared Error:", mse) #printing the Mean Squared Error

print("\nx_test:\n", x_test) #printing the test data for independent variable
print("\ny_test (actual):\n", y_test) #printing the actual values of the dependent variable for the test data
print("\ny_pred (predicted):\n", y_pred) #printing the predicted values of the dependent variable for the test data 

#Visualizing the results
plt.scatter(x_test, y_test, color = "blue", label = "actual") #scatter plot for actual values
plt.scatter(x_test, y_pred, color = "red", label = "predicted")
plt.xlabel("Rainfall")
plt.ylabel("Flood")
plt.title("Model Evaluation: Actual vs Predicted")
plt.plot(x_test, y_pred, color='orange', label='predicted') #plotting the regression line for predicted values
plt.legend() #adding legend to the plot 
plt.savefig("outputs/day03_evaluation.png") #saving the plot as an image in outputs folder
plt.show() #displaying the plot