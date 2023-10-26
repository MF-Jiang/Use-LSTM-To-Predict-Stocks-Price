# Use-LSTM-To-Predict-Stocks-Price
This project uses Long Short-Term Memory (LSTM) networks to predict stock price trends. The project includes steps for data acquisition, preprocessing, model training, and prediction. You can use this project to study and predict stock price trends.

## Usage

### Data Acquisition

Set the time range for stock data retrieval in the `start` and `end` variables. Then, use Pandas DataReader to fetch the data.
```python
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2023, 9, 30)
df = web.DataReader('GOOGL', 'stooq', start, end)
```

### Data Preprocessing
Use the Stock_Price_LSTM_Data_Precessing function to preprocess the data, including data standardization and feature construction.
```python
X, y, X_lately = Stock_Price_LSTM_Data_Precessing(df, mem_his_days, pre_days)
```

### Model Training
Train the model with different parameters, including mem_days, lstm_layers, dense_layers, and units. Model training uses the TensorFlow and Keras libraries.
```python
model = Sequential()
# Add layers and configure
model.compile(optimizer='adam', loss='mse', metrics=['mape'])
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint])
```

### Model Evaluation and Prediction
Load the trained model, evaluate its performance, and make predictions using the model.
```python
best_model = load_model('./models/5.48_01men_5_lstm_1_dense_2_unit_32')
best_model.summary()
best_model.evaluate(X_test, y_test)
pre = best_model.predict(X_test)
```

### Visualization of Results
Use Matplotlib to visualize actual stock prices and model prediction results.
```python
plt.plot(df_time, y_test, color='red', label='price')
plt.plot(df_time, pre, color='blue', label='predict')
plt.show()
```

### Configuration Parameters
In the project, you can adjust different parameters to train models with various configurations, such as mem_days, lstm_layers, dense_layers, and units.

## Project Structure
The project includes the following files and folders:

- Stock Price Prediction.ipynb: Jupyter Notebook file containing code and demonstrations.
- models/: Folder for saving the best models.
- Other Python scripts and library files.

## Dependencies
The project depends on the following Python libraries:

- pandas_datareader
- scikit-learn
- numpy
- tensorflow
- matplotlib
You can install these libraries using pip.
```
pip install pandas_datareader scikit-learn numpy tensorflow matplotlib
```

