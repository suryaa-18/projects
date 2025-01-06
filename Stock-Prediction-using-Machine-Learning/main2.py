import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

yf.pdr_override()

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val

def inverse_transform(data, min_val, max_val):
    print(data)
    return data * (max_val - min_val) + min_val

def mean_squared_error_custom(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

tech_list = ['AAPL']
end = datetime.now()
start = datetime(end.year, end.month-6, end.day)

company_list = []
for stock in tech_list:
    company_data = yf.download(stock, start, end)
    company_data['company_name'] = stock
    company_list.append(company_data)

df = pd.concat(company_list)
df.reset_index(inplace=True)
print()
print()
print()
print()

dwnlddata = 'stockdata.csv'
df.to_csv(dwnlddata, index=False)
print("Downloaded stock data saved to a file")
print()
data = df['Close'].values.reshape(-1, 1)

data_normalized, data_min, data_max = normalize_data(data)

def split_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

train_data, test_data = split_data(data_normalized)

def create_dataset(data, time_steps=10):
    x=[]
    y=[]
    for i in range(len(data) - time_steps):
        x.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

timesteps = 10
X_train, y_train = create_dataset(train_data, timesteps)
X_test, y_test = create_dataset(test_data, timesteps)

rf = RandomForestRegressor(n_estimators=10)
lr = LinearRegression()
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
svr.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
pred_lr = lr.predict(X_test)
pred_svr = svr.predict(X_test)

pred_rf = inverse_transform(pred_rf, data_min, data_max)
pred_lr = inverse_transform(pred_lr, data_min, data_max)
pred_svr = inverse_transform(pred_svr, data_min, data_max)
y_test_inv = inverse_transform(y_test, data_min, data_max)

mse_rf = mean_squared_error_custom(y_test_inv, pred_rf)
mse_lr = mean_squared_error_custom(y_test_inv, pred_lr)
mse_svr = mean_squared_error_custom(y_test_inv, pred_svr)

print("Random Forest Custom MSE:", mse_rf)
print("Linear Regression Custom MSE:", mse_lr)
print("SVR Custom MSE:", mse_svr)

predictions_df = pd.DataFrame({
    'Actual Price': y_test_inv.flatten(),
    'Random Forest Prediction': pred_rf.flatten(),
    'Linear Regression Prediction': pred_lr.flatten(),
    'SVR Prediction': pred_svr.flatten()
})
print()
print()
output_file_predictions = 'output.csv'
predictions_df.to_csv(output_file_predictions, index=False)
print("Predicted stock prices saved to output file")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(pred_rf, label='Random Forest Prediction')
plt.plot(pred_lr, label='Linear Regression Prediction')
plt.plot(pred_svr, label='SVR Prediction')
plt.legend()
plt.show()
