import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to plot loss based on what Tensorflow outputs
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss') # Plot loss
  plt.plot(history.history['val_loss'], label='val_loss') # Plot validation loss
  plt.xlabel('Epoch') # Label the x-axis
  plt.ylabel('Error') # Label the y-axis
  plt.legend()
  plt.grid(True)

# Firstly read in the data from the csv file
df = pd.read_csv("datapost.csv")
df.head() # Remove the labels from the data

# Split the data into training and testing sets
train_dataset = df.sample(frac=0.8, random_state=0) # 80% of the data for training
test_dataset = df.drop(train_dataset.index) # The remaining 20% for testing

# Seperate the labels used for prediction later on in the program
shares = test_dataset.pop("shares")
ticker = test_dataset.pop("ticker")

# Delete the labels that aren't used in the training set
train_dataset.pop("shares")
train_dataset.pop("ticker")
train_features = train_dataset.copy()
test_features = test_dataset.copy()

# Remove the cap from the labels, as the correct answer should (obviously) not be used to train the model
train_labels = train_features.pop("cap")
test_labels = test_features.pop("cap")

# Print the data so we can see what we're working with - mainly for testing purposes but helpful for visualising the data
print(train_features)

# Normalise the data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# Create the model with a diamond shape
model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1),
])

# Optimiser to use with MEA for error, as it is helpful to know the average error in USD rather than something unreadable as to better understand what kind of error is achievable
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss = 'mean_absolute_error',
)

# Give a summary, again to help with debugging and understanding the code
model.summary()

# Train the model, storing the history of loss so we can plot it later on (so we can see how the model is doing and prevent overfitting)
history = model.fit(train_features, train_labels, epochs=350, validation_split=0.2)

# Plot the loss
plot_loss(history)
plt.show()

# Use the model to predict the market cap
test_predictions = model.predict(np.array(test_features)).flatten()

# Plot the actual vs predicted market cap
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')

# Turn these to numpy arrays so we can use them in the next step
ticker.to_numpy()
test_labels.to_numpy()

# Output the predictions, along with the prediction discrepancy as a decimal
for i in range(0,len(test_labels)-1):
    print(ticker.to_numpy()[i], test_labels.to_numpy()[i]/shares.to_numpy()[i], test_predictions[i]/shares.to_numpy()[i], (test_predictions[i]/shares.to_numpy()[i])/(test_labels.to_numpy()[i]/shares.to_numpy()[i]))

## Output example ##
# AAPL 135 160 1.18518519

# This output is a prediction of the market cap of Apple Inc.
# The actual price is 135, the predicted price is 160, and the discrepancy is 1.18518519