import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

# Drop rows with missing values
dataset = raw_dataset.dropna()

# Split features and target variable
X = dataset.drop('MPG', axis=1)
y = dataset['MPG']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model architecture
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)  # Output layer with 1 neuron and no activation function (linear activation)
    ])
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

# Build the model
input_shape = (X_train_scaled.shape[1],)  # Shape of input features
model = build_model(input_shape)

# Display model summary
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model on test data
loss, mae, mse = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Mean Absolute Error on test data: {mae}")

# Make predictions
predictions = model.predict(X_test_scaled)

# Display predicted MPG values
print("Predicted MPG values:")
for pred in predictions:
    print(pred[0])
