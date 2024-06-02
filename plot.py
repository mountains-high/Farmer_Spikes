import pandas as pd
import matplotlib.pyplot as plt

#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.size'] = 10

# Load data from CSV files
train_loss_snn = pd.read_csv('Training_loss_SNNs.csv', header=None)
train_loss_lstm = pd.read_csv('Training_loss_LSTM.csv', header=None)
val_loss_snn = pd.read_csv('Validation_loss_SNNs.csv', header=None)
val_loss_lstm = pd.read_csv('Validation_loss_LSTM.csv', header=None)

fig = plt.figure(figsize=(10, 6))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))

# Training Loss
ax1.plot(train_loss_snn.index, train_loss_snn[0], label='SNNs')
ax1.plot(train_loss_lstm.index, train_loss_lstm[0], label='LSTM')
ax1.set_ylabel("Training Loss")
#ax1.set_title("Performance Metrics")
ax1.set_xlabel("Epochs")
ax1.grid()
ax1.legend()

# Validation Loss
ax2.plot(val_loss_snn.index, val_loss_snn[0], label='SNNs')
ax2.plot(val_loss_lstm.index, val_loss_lstm[0], label='LSTM')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Validation Loss")
ax2.grid()
ax2.legend()



plt.tight_layout()
plt.savefig('losses.png', dpi=300, bbox_inches='tight')
plt.show()