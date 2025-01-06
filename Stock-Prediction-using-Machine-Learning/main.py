import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import yfinance as yf

yf.pdr_override()  

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']  
end = datetime.now() 
start = datetime(end.year, end.month-1, end.day) 

company_list = [] 
for stock in tech_list: 
    company_data = yf.download(stock, start, end) 
    company_data['company_name'] = stock 
    company_list.append(company_data) 
print(company_list)

# Concatenate DataFrames 
df = pd.concat(company_list) 
 
# Reset index to make 'Date' a regular column 
df.reset_index(inplace=True) 

data = df['Close'].values.reshape(-1, 1) 

# Normalize the data 
scaler = MinMaxScaler(feature_range=(0, 1)) 
data_normalized = scaler.fit_transform(data) 

# Step 2: Data Splitting 
train_size = int(len(data_normalized) * 0.8) 
test_size = len(data_normalized) - train_size 
train_data, test_data = data_normalized[0:train_size], data_normalized[train_size:len(data_normalized)]

# Step 3: Prepare data for LSTM 
def create_dataset(data, time_steps): 
    X, y = [], [] 
    for i in range(len(data) - time_steps): 
        X.append(data[i:(i + time_steps), 0]) 
        y.append(data[i + time_steps, 0]) 
    return np.array(X), np.array(y) 

time_steps = 10 
X_train, y_train = create_dataset(train_data, time_steps) 
X_test, y_test = create_dataset(test_data, time_steps) 

# Reshape input to be [samples, time steps, features] 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 

# Step 4: Model Definition 
model = Sequential([ 
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)), 
    Dropout(0.2), 
    LSTM(units=50, return_sequences=True), 
    Dropout(0.2), 
    LSTM(units=50), 
    Dropout(0.2), 
    Dense(units=1) 
]) 

# Step 5: Model Compilation 
optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss=MeanSquaredError())  

# Step 6: Model Training 
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1) 

# Step 7: Model Evaluation 
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.legend() 
plt.show() 

# Predictions 
predictions = model.predict(X_test) 
predictions = scaler.inverse_transform(predictions) 

# Visualize predictions vs actual values 
plt.plot(predictions, label='Predicted') 
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual') 
plt.legend() 
plt.show()