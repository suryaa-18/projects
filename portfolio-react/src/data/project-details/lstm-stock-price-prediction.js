export default {
  architecture: [
    'Data Collection & Preprocessing: Historical stock price data is automatically downloaded from Yahoo Finance using the yfinance API for the selected company. Closing prices are extracted, normalized using Min-Max scaling, and transformed into fixed-length time-series sequences using a sliding window approach (10 previous trading days) to capture short-term market trends. The dataset is then split into training and testing subsets for model evaluation.',

    'Machine Learning Model Training: Three regression models—Random Forest Regressor, Linear Regression, and Support Vector Regression (RBF kernel)—are independently trained on the generated time-series features. Each model learns the relationship between historical closing prices and the next-day stock price using supervised learning, enabling a comparative analysis of tree-based, linear, and kernel-based regression techniques.',

    'Prediction & Performance Evaluation: The trained models generate next-day stock price predictions on unseen test data. Predicted values are inverse transformed to the original price scale before evaluation. Model performance is assessed using Mean Squared Error (MSE), and actual versus predicted stock prices are exported to CSV and visualized using Matplotlib to facilitate performance comparison and trend analysis.'
  ],

  result:
    'Successfully implemented an end-to-end stock price prediction pipeline that automatically collects financial market data, trains multiple regression models, and compares their predictive performance. The system generates next-day stock price forecasts, exports prediction results, computes Mean Squared Error (MSE) for each model, and visualizes actual versus predicted price movements for comparative analysis.',

  novelty:
    'Developed a unified forecasting framework that automates the complete machine learning workflow—from real-time financial data acquisition and preprocessing to model benchmarking and visualization. The project enables side-by-side comparison of ensemble (Random Forest), statistical (Linear Regression), and kernel-based (SVR) regression models under a common evaluation pipeline, making it easy to analyze the strengths and limitations of different prediction approaches.'
};