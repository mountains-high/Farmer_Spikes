import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Loading the actual and predicted values from their respective CSV files
actual_data = pd.read_csv('scaled_actual_power_outputs.csv', header=None, nrows=300)
predicted_data = pd.read_csv('scaled_predicted_lstm_power_outputs.csv', header=None, nrows=300)

# data is in a single column format
actual = actual_data[0].values
predicted = predicted_data[0].values

# Plotting
fig, ax = plt.subplots()

# Plot actual values
ax.plot(actual, "--", label="Actual")

# Plot predicted values
ax.plot(predicted, label="Predicted")

# Set labels and title
ax.set_xlabel("Data Points")
ax.set_ylabel("Scaled Power Output")
ax.legend()

# Save and show the plot
plt.savefig('Scaled_Prediction_comparison.png')
plt.show()
