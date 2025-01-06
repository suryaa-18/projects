import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load Data
company = input("Enter the company name (file name without .csv): ")
path = "D:\\VIT\\Sem 1\\Python\\Stock-prediction-main\\Stock-prediction-main\\stockPredictionProject\\" + company + ".csv"
print("Loading data from:", path)

# Step 2: Read CSV and Preprocess
df = pd.read_csv(path)
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime format
df.drop('Adj Close', axis=1, inplace=True)  # Drop 'Adj Close' column as it's not used

# Convert 'Volume' to float (if needed)
df['Volume'] = df['Volume'].astype(float)

# Remove any rows with NaN or infinite values
df = df[np.isfinite(df).all(1)]

# Plotting the historical 'Open' prices
df['Open'].plot(figsize=(16, 6), title=f"{company} - Open Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.show()

# Step 3: Prepare Data for Training
# Features: Open, High, Low, Volume; Target: Close
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Step 4: Split the data into Training and Testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Initialize and Train the Model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Step 6: Make Predictions
predicted = regressor.predict(x_test)

# Calculate Mean Squared Error for Model Evaluation
mse = mean_squared_error(y_test, predicted)
print(f"Mean Squared Error: {mse}")

# Step 7: Create DataFrame for Comparison of Actual vs Predicted Values
dfr = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})
print("\nActual vs Predicted Stock Closing Prices:")
print(dfr.head())

# Step 8: Plot Actual and Predicted Prices
plt.figure(figsize=(12, 6))

# Plot actual values
plt.plot(dfr.index, dfr['Actual'], label="Actual Closing Price", color="blue")

# Plot predicted values
plt.plot(dfr.index, dfr['Predicted'], label="Predicted Closing Price", color="orange")

# Add labels, title, and legend
plt.xlabel("Index")
plt.ylabel("Stock Closing Price")
plt.title(f"{company} - Actual vs Predicted Stock Closing Prices")
plt.legend()

# Display the plot
plt.show()

# Step 9: Model Score
score = regressor.score(x_test, y_test)
print(f"Model R^2 Score: {score}")
