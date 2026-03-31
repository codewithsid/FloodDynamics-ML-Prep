import pandas as pd
import matplotlib.pyplot as plt

#data dictionary (keys and array of values)
data = {
    "rainfall": [5, 15, 25, 35, 45, 60], 
    "flood": [0, 0, 1, 1, 1, 1]
}

#making dataframe from data dictionary
df = pd.DataFrame(data)

#inspecting data
print("\nDataset:\n", df)
print("\nStats:\n", df.describe()) #describing data (count, mean, std, min, 25%, 50%, 75%, max)

#showing data with Matplotlib
plt.scatter(df["rainfall"], df["flood"], color='blue')
plt.xlabel("Rainfall")
plt.ylabel("Flood")
plt.title("Rainfall vs Flood Occurence")
plt.show() #displaying plot
plt.savefig("outputs/day01_scatter.png") #saving the plot as an image in outputs folder