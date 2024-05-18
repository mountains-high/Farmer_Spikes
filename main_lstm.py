import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate
from snntorch import utils

# use GPU if available
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

# path to raw weather data
DATA_PATH = "./data/weather_rawdata/"
# Near Des Moines, Iowa
LOCATION_PREFIX = "42.02124491636418_-93.77372190333062_42.0212_-93.7741_psm3_60_"

INPUT_COLUMNS = ["Month", "Hour", "DNI", "DHI", "GHI", "Dew Point", "Temperature", "Pressure",
                 "Relative Humidity", "Wind Direction", "Wind Speed", "Surface Albedo",]
OUTPUT_COLUMNS = ["Power Next"]

# number of training data points
N_TRAIN_HOURS = 365 * 24 * 18
# number of validation data points
N_VAL_HOURS = 365 * 24 * 2

# list of all filepaths
filepaths = []
for year in range(2000, 2021):
  path = os.path.join(DATA_PATH, f"{LOCATION_PREFIX}{str(year)}.csv")
  filepaths.append(path)
  
  
  weather_data = []

for path in filepaths:
  weather_data_year = pd.read_csv(path, skiprows=2)
  weather_data.append(weather_data_year)
  
weather_data = pd.concat(weather_data)
weather_data = weather_data.reset_index(drop=True)

# Add timestamp
weather_data["Timestamp"] = pd.to_datetime(weather_data[["Year", "Month", "Day", "Hour", "Minute"]])
weather_data = weather_data.drop(["Year", "Day", "Minute"], axis=1)
# reorder columns
weather_data = weather_data[["Timestamp", "Month", "Hour", "DNI", "DHI", "GHI", "Dew Point", "Temperature", "Pressure",
                             "Relative Humidity", "Wind Direction", "Wind Speed", "Surface Albedo",]]



import PySAM.Pvwattsv8 as pv
import PySAM.Grid as gr
import PySAM.Utilityrate5 as ur
import PySAM.Singleowner as so

output_power = []

for path in tqdm(filepaths): 
  # create an instance of the Pvwattsv8 module with defaults from the PVWatts - Single Owner configuration
  system_model = pv.default('PVWattsSingleOwner')
  # create instances of the other modules with shared data from the PVwattsv8 module
  grid_model = gr.from_existing(system_model, 'PVWattsSingleOwner')
  utilityrate_model = ur.from_existing(system_model, 'PVWattsSingleOwner')
  financial_model = so.from_existing(system_model, 'PVWattsSingleOwner')
  system_model.SolarResource.solar_resource_file = path
  # run the modules in the correct order
  system_model.execute()
  grid_model.execute()
  utilityrate_model.execute()
  financial_model.execute()
  # display results
  #print( 'Annual AC Output in Year 1 = {:,.3f} kWh'.format( system_model.Outputs.ac_annual ) )
  #print( 'Net Present Value = ${:,.2f}'.format(financial_model.Outputs.project_return_aftertax_npv) )
  dc = system_model.Outputs.dc
  #ac = system_model.Outputs.ac
  
  output_power.append(dc)

# concat individual lists
output_power = np.concatenate(output_power)
# create dataframe
output_power = pd.DataFrame(output_power, columns=["Power"])
# shift to align current weather data to power of the next day
output_power["Power Next"] = output_power["Power"].shift(-1)


combined_data = pd.concat([weather_data, output_power], axis=1)
combined_data = combined_data.dropna()
combined_data = combined_data.set_index("Timestamp")
print(combined_data.head(10))
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas



first_year = combined_data[combined_data.index < pd.to_datetime("2000-02-01")]
first_year.plot(
  subplots=True,
  #title="PV Panel Weather and Power Data",
  grid=True,
  layout=(7,2),
  figsize=(10,12)
  )
plt.tight_layout()
plt.savefig('pv_panel_weather_and_power_data.pdf')
plt.show()

BATCH_SIZE = 72

# convert to numpy arrays
input_series = combined_data[INPUT_COLUMNS].to_numpy()
output_series = combined_data[OUTPUT_COLUMNS].to_numpy()

input_scaler = StandardScaler()
output_scaler = StandardScaler()

input_series = input_scaler.fit_transform(input_series)
output_series = output_scaler.fit_transform(output_series)

input_series = torch.as_tensor(input_series).to(torch.float32).to(device)
output_series = torch.as_tensor(output_series).to(torch.float32).to(device)

# training data
train_inputs = input_series[:N_TRAIN_HOURS, :]
train_outputs = output_series[:N_TRAIN_HOURS]

# create dataloaders
train_dataset = TensorDataset(train_inputs, train_outputs)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# validation data
val_inputs = input_series[N_TRAIN_HOURS:N_TRAIN_HOURS+N_VAL_HOURS, :]
val_outputs = output_series[N_TRAIN_HOURS:N_TRAIN_HOURS+N_VAL_HOURS]

val_dataset = TensorDataset(val_inputs, val_outputs)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# test data
test_inputs = input_series[N_TRAIN_HOURS+N_VAL_HOURS:, :]
test_outputs = output_series[N_TRAIN_HOURS+N_VAL_HOURS:]

test_dataset = TensorDataset(test_inputs, test_outputs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



#
# Create network
#

class LSTM(nn.Module):
  def __init__(self, input_size):
    """LSTM NN Constructor"""
    super(LSTM, self).__init__()
    
    self.lstm = nn.LSTM(input_size, 50)
    self.fc = nn.Linear(50, 1)
    
  def forward(self, x):
    """Forward pass of LSTM network""" 
    
    lstm_out, _ = self.lstm(x)
    output = self.fc(lstm_out)
    return output

# load onto GPU
lstm_net = LSTM(len(train_inputs[0])).to(device)


#
# Train Network
#

# default learning rate from tf.keras.optimizers.Adam
LEARNING_RATE = 0.0001

EPOCHS = 100

# Adam optimizers
optimizer = optim.Adam(lstm_net.parameters(), lr=LEARNING_RATE)
# Mean absolute error
loss_fn = nn.L1Loss()


loss_train_hist = []
loss_val_hist = []


# training loop
for epoch in tqdm(range(EPOCHS)):
  loss_train_epoch = []

  lstm_net.train()
  for inputs, outputs in train_loader:
    # forward pass
    predictions = lstm_net(inputs)
    # calculate loss from membrane potential at last timestep 
    loss_val = loss_fn(predictions, outputs)
    # zero out gradients
    optimizer.zero_grad() 
    # calculate gradients
    loss_val.backward() 
    # update weights
    optimizer.step() 

    # store loss
    loss_train_epoch.append(loss_val.item())

  # calculate average loss p/epoch
  avg_loss_epoch_train = sum(loss_train_epoch) / len(loss_train_epoch) 
  loss_train_hist.append(avg_loss_epoch_train)
  
  
  loss_val_epoch = []
  
  lstm_net.eval()
  for inputs, outputs in val_loader:
    predictions = lstm_net(inputs)
    loss_val = loss_fn(predictions, outputs)
    loss_val_epoch.append(loss_val.item())
    
  avg_loss_epoch_val = sum(loss_val_epoch) / len(loss_val_epoch)
  loss_val_hist.append(avg_loss_epoch_val)
  
  print(f"Epoch: {epoch+1}/{EPOCHS}, Train Loss: {avg_loss_epoch_train}, Val Loss: {avg_loss_epoch_val}") 
  
  
# Plot of train and loss functions

plt.figure()

plt.subplot(2,1,1)
plt.plot(loss_train_hist)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel("Training Loss")
plt.title("Performance Metrics")
plt.grid()

plt.subplot(2,1,2)
plt.plot(loss_val_hist)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.grid()
plt.tight_layout()
plt.savefig('Validation_loss')
plt.show()

# Run on test data

lstm_net.eval()
with torch.no_grad():
  predictions = lstm_net(test_inputs)
  
#predictions = predictions.cpu()
#predictions = output_scaler.inverse_transform(predictions)

fix, ax = plt.subplots()

ax.plot(output_scaler.inverse_transform(test_outputs.cpu()[0:720]), "--", label="Actual")
ax.plot(output_scaler.inverse_transform(predictions.cpu()[0:720]), label="Predicted")

ax.set_xlabel("Hours")
ax.set_ylabel("Power Output (W)")

ax.legend()
ax.grid()
plt.savefig('Prediction')
plt.show()

#===========================================================
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Extract the first 720 values for predictions and actual outputs
predictions_720 = predictions.cpu().numpy()[0:720]
actual_720 = test_outputs.cpu().numpy()[0:720]

# Initialize separate MinMaxScalers for actual and predicted
scaler_actual = MinMaxScaler(feature_range=(0, 1))
scaler_predicted = MinMaxScaler(feature_range=(0, 1))

# Scale the actual and predicted values separately
actual_scaled = scaler_actual.fit_transform(actual_720)
predictions_scaled = scaler_predicted.fit_transform(predictions_720)

# Round the predicted power output to keep only two values after the decimal point
predictions_scaled = np.round(predictions_scaled, 2)
# Create DataFrames from the scaled arrays
df_actual = pd.DataFrame(actual_scaled, columns=['Actual Power Output'])
df_predicted = pd.DataFrame(predictions_scaled, columns=['Predicted Power Output'])

# Save the DataFrames to separate CSV files
df_actual.to_csv('scaled_actual_power_outputs.csv', index=False)
df_predicted.to_csv('scaled_predicted_lstm_power_outputs.csv', index=False)

print("Scaled actual power outputs saved to 'scaled_actual_power_outputs.csv'.")
print("Scaled predicted power outputs saved to 'scaled_predicted_lstm_power_outputs.csv'.")
