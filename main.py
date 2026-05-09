print("Code is running...")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Create outputs folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Load dataset (CORRECT NAME)
df = pd.read_csv("electricity.csv")

print("\nDataset Preview:")
print(df.head())

# Features and target
X = df[['temperature', 'hour', 'previous_usage']]
y = df['consumption']

# Model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

print("\nPredicted Values:")
print(predictions)

# ---------------- GRAPH 1 ----------------
plt.figure()
plt.plot(df['hour'], y)
plt.title("Electricity Consumption Over Time")
plt.xlabel("Hour")
plt.ylabel("Consumption")
plt.savefig("outputs/graph1.png")
plt.show()

# ---------------- GRAPH 2 ----------------
plt.figure()
plt.scatter(df['temperature'], y)
plt.title("Temperature vs Consumption")
plt.xlabel("Temperature")
plt.ylabel("Consumption")
plt.savefig("outputs/graph2.png")
plt.show()

# ---------------- GRAPH 3 ----------------
plt.figure()
plt.plot(df['hour'], y, label="Actual")
plt.plot(df['hour'], predictions, linestyle='dashed', label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Consumption")
plt.xlabel("Hour")
plt.ylabel("Consumption")
plt.savefig("outputs/graph3.png")
plt.show()

# ---------------- TEST ----------------
new_data = [[34, 14, 140]]
predicted_value = model.predict(new_data)

print("\nPredicted consumption:", predicted_value[0])