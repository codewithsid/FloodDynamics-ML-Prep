import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
#dataset creation
data = {
    "rainfall": [5, 15, 25, 35, 45, 60], 
    "flood": [0, 0, 1, 1, 1, 1] 
}

df = pd.DataFrame(data)

#Processing Data for Model
x = df[["rainfall"]] #independent variable (2D array) Double brackets to keep it as a DataFrame
y = df["flood"] #dependent variable (1D array) Single brackets to get a Series

model = LinearRegression() #creating linear regression model
model.fit(x, y)#telling model to learn from data
predictions = model.predict(x) #making predictions using the trained model

#Visualizing the results
plt.scatter(x, y) 
plt.plot(x, predictions, color='red') #plotting the regression line
plt.xlabel("Rainfall")
plt.ylabel("Flood")
plt.title("Linear Regression: Rainfall vs Flood Occurrence")
plt.savefig("outputs/day02_regression.png") #saving the plot as an image in outputs folder

#Details of the Model
print("Slope (m):", model.coef_) #slope of the line
print("Intercept (b):", model.intercept_) #y-intercept of the line

plt.show() #displaying the plot