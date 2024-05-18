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

# Define model
#
class SNN(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, in_features, hidden):
        super().__init__()

        self.timesteps = timesteps # number of time steps to simulate the network
        self.hidden = hidden # number of hidden neurons
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function

        # randomly initialize decay rate and threshold for layer 1
        beta_in = torch.rand(self.hidden)
        thr_in = torch.rand(self.hidden)

        # layer 1
        self.fc_in = torch.nn.Linear(in_features=in_features, out_features=self.hidden)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(self.hidden)
        thr_hidden = torch.rand(self.hidden)

        # layer 2
        self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(1)

        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
        self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=1)
        self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")

    def forward(self, x):
        """Forward pass for several time steps."""

        # Initalize membrane potential
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_3 = self.li_out.init_leaky()

        # Empty lists to record outputs
        mem_3_rec = []

        # Loop over
        for step in range(self.timesteps):
            cur_in = self.fc_in(x)
            spk_in, mem_1 = self.lif_in(cur_in, mem_1)

            cur_hidden = self.fc_hidden(spk_in)
            spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

            cur_out = self.fc_out(spk_hidden)
            _, mem_3 = self.li_out(cur_out, mem_3)

            mem_3_rec.append(mem_3)

        return torch.stack(mem_3_rec)

# Parameters
TIMESTEPS = 20
HIDDEN_LAYERS = 128

snn_net = SNN(
  timesteps=TIMESTEPS,
  hidden=HIDDEN_LAYERS,
  in_features=len(train_inputs[0])
  )
snn_net = snn_net.to(device)


#
# Output without any training 
#

# run a single forward-pass to see what output is
with torch.no_grad():
  for inputs, outputs in train_loader:
    mem = snn_net(inputs)

# record outputs for later plotting
sample_outputs = outputs
sample_mem = mem


#
# Training loop
#

# hyperparameters
EPOCHS = 100
LEARNING_RATE = 1e-3

# optimizer
optimizer = torch.optim.Adam(params=snn_net.parameters(), lr=1e-3)
# loss function for membrane potential
loss_function = torch.nn.MSELoss()

# list to store loss at each timestep
loss_train_hist = [] 
loss_val_hist = []

# training loop
for epoch in tqdm(range(EPOCHS)):
  loss_train_epoch = []

  snn_net.train()
  for inputs, outputs in train_loader:
    # forward pass
    mem = snn_net(inputs)
    # calculate loss from membrane potential at last timestep 
    loss_val = loss_function(mem[-1,:,:], outputs)
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
  
  snn_net.eval()
  for inputs, outputs in val_loader:
    mem = snn_net(inputs)
    loss_val = loss_function(mem[-1,:,:], outputs)
    loss_val_epoch.append(loss_val.item())
    
  avg_loss_epoch_val = sum(loss_val_epoch) / len(loss_val_epoch)
  loss_val_hist.append(avg_loss_epoch_val)
  
  print(f"Epoch: {epoch+1}/{EPOCHS}, Train Loss: {avg_loss_epoch_train}, Val Loss: {avg_loss_epoch_val}") 
  
  
  # Plot of network before training

fig, ax = plt.subplots()
# plot expected output
ax.plot(sample_outputs.squeeze(1).cpu(), '--', label="Target")
# plot first 5 membrane potential outputs
for idx in range(0, min(TIMESTEPS, 5)):
  ax.plot(sample_mem[idx,:,0].cpu(), alpha=0.6)
  
ax.set_title("Untrained Output Neuron")
ax.set_xlabel("Time")
ax.set_ylabel("Membrane Potential")
ax.legend(loc='best')
plt.show()

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
plt.savefig('Validation Loss SNNs')
plt.show()


# Run on test data

snn_net.eval()
with torch.no_grad():
  predictions = snn_net(test_inputs)
  
#predictions = predictions.cpu()
#predictions = output_scaler.inverse_transform(predictions)

fix, ax = plt.subplots()

ax.plot(output_scaler.inverse_transform(test_outputs.cpu()[0:720]), "--", label="Actual")
ax.plot(output_scaler.inverse_transform(predictions.cpu()[0,0:720,:]), label="Predicted", alpha=0.6, color='green')

ax.set_xlabel("Hours")
ax.set_ylabel("Power Output (W)")

ax.legend()
ax.grid()
plt.savefig('snntorch_result')
plt.show()


#========================================================
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Assuming predictions is a PyTorch tensor with the shape [1, n_predictions, n_features]
# For simplicity and clarity, the code will be adjusted to handle common shapes directly

# Convert PyTorch tensors to numpy and remove unnecessary dimensions
# Ensure the data is 2D: (720, number_of_features), here assuming 1 feature for simplicity
predictions_720 = predictions.cpu().numpy()[0, :720].reshape(-1, 1)
actual_720 = test_outputs.cpu().numpy()[:720].reshape(-1, 1)

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the actual values
actual_720_scaled = scaler.fit_transform(actual_720)

# Reset the scaler for predictions to ensure independent scaling
scaler = MinMaxScaler(feature_range=(0, 1))
predictions_720_scaled = scaler.fit_transform(predictions_720)

# Round the predicted power output to keep only two decimal points
predictions_720_scaled_rounded = np.round(predictions_720_scaled, 2)

# Create DataFrames from the scaled arrays
actual_df = pd.DataFrame(actual_720_scaled, columns=OUTPUT_COLUMNS)
predicted_df = pd.DataFrame(predictions_720_scaled_rounded, columns=['Predicted Power Output'])

# Save the DataFrames to separate CSV files
actual_df.to_csv('actual_snn_power_outputs_first_720_scaled.csv', index=False)
predicted_df.to_csv('predicted_power_outputs_snn_first_720_scaled.csv', index=False)

print("Actual power outputs (scaled) saved to 'actual_power_outputs_first_720_scaled.csv'.")
print("Predicted power outputs (scaled and rounded) saved to 'predicted_power_outputs_snn_first_720_scaled.csv'.")
