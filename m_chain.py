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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

first_year = combined_data[combined_data.index < pd.to_datetime("2000-02-01")]
first_year.plot(
  subplots=True,
  title="PV Panel Weather and Power Data",
  grid=True,
  layout=(7,2),
  figsize=(10,12)
  )
plt.savefig('pv_panel_weather_and_power_data')
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

# Markov Chain Implementation
#=========================================================================================
# Step 1: Define states for the Markov Chain by discretizing 'Power Next'
power_bins = pd.qcut(combined_data['Power Next'], q=4, labels=False, duplicates='drop')
combined_data['Power State'] = power_bins

# Initialize a transition matrix assuming 4 discrete states
transition_matrix = np.zeros((4, 4))

# Step 2: Fill the transition matrix with counts
for i in range(len(power_bins) - 1):
    current_state = power_bins.iloc[i]
    next_state = power_bins.iloc[i + 1]
    transition_matrix[current_state, next_state] += 1

# Step 3: Normalize the transition matrix to convert counts to probabilities
row_sums = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)

# Prediction using the transition matrix
#-----------------------------------------------------------------------------------------
# Prepare the starting point for prediction
initial_state = combined_data['Power State'][N_TRAIN_HOURS+N_VAL_HOURS]
predicted_states = [initial_state]  # Initialize with the first state of the test set

# Predict the next 719 states
for _ in range(1, 720):  # 720 including the initial state
    current_state = predicted_states[-1]
    next_state = np.argmax(transition_matrix[current_state])
    predicted_states.append(next_state)

# Map predicted states to their median 'Power Next' values for visualization
# Assuming you've determined median values for each state
state_to_power_values = combined_data.groupby('Power State')['Power Next'].median().values
predicted_power_output = [state_to_power_values[state] for state in predicted_states]

# Extract actual 'Power Next' values for the first 720 hours of the test dataset
actual_power_output = combined_data['Power Next'][N_TRAIN_HOURS+N_VAL_HOURS:N_TRAIN_HOURS+N_VAL_HOURS+720].values

# Visualization
#-----------------------------------------------------------------------------------------
plt.figure(figsize=(15, 5))
plt.plot(actual_power_output, label="Actual Power Output", marker='.', linestyle='-', markersize=2, linewidth=0.5)
plt.plot(predicted_power_output, label="Predicted Power Output", marker='x', linestyle='--', markersize=2, linewidth=0.5)
plt.title("Actual vs. Predicted Power Output for the First 720 Hours")
plt.xlabel("Hour")
plt.ylabel("Power Output")
plt.legend()
plt.tight_layout()
plt.savefig('MarkovChain_Prediction_Comparison.png')
plt.show()


