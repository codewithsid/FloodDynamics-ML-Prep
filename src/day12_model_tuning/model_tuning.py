import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/flood_data_harder.csv")

x = df[["rainfall", "temperature", "soil_moisture"]]
y = df["flood"]

logistic_scores = []
decision_tree_scores = [] 

#Looping over for multiple splits
for i in range(20) :
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

    #logistic Regression
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    log_model = LogisticRegression()
    log_model.fit(x_train_scaled, y_train)

    y_pred_log = log_model.predict(x_test_scaled)
    log_acc = accuracy_score(y_test, y_pred_log)

    logistic_scores.append(log_acc)

    #Decision Tree
    tree_model = DecisionTreeClassifier(max_depth = 2)
    tree_model.fit(x_train, y_train)

    y_pred_tree = tree_model.predict(x_test)
    tree_acc = accuracy_score(y_test, y_pred_tree)

    decision_tree_scores.append(tree_acc)

#Results
print("\nAverage Logistic Regression Accuracy:", np.mean(logistic_scores))
print("\nAverage Decision Tree Accuracy:", np.mean(decision_tree_scores))

#changing depths 
depths = [1, 2, 3, 5, 10]

for depth in depths:
    scores = []
    train_scores = []
    test_scores = []

    for i in range(20):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

        model = DecisionTreeClassifier(max_depth = depth)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        accuarcy = accuracy_score(y_test, y_pred)

        train_acc = accuracy_score(y_train, model.predict(x_train))
        test_acc = accuracy_score(y_test, model.predict(x_test))

        scores.append(accuarcy)
        train_scores.append(train_acc)
        test_scores.append(test_acc)

    avg_score = np.mean(scores)
    avg_train = np.mean(train_scores)
    avg_test = np.mean(test_scores)
    print(f"Depth: {depth}, {avg_score}")
    print(f"Depth: {depth}, Train: {avg_train}, Test: {avg_test}")

plt.figure(figsize = (10, 6))
plot_tree(tree_model, feature_names = x.columns, filled = True)
plt.title("Decision Tree Visualization")
plt.savefig("outputs/decision_tree.png")
plt.show() 