import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
#input_size = 784 # 28x28
hidden_size = 64
num_classes = 2
num_epochs = 10
batch_size = 600
learning_rate = 1e-4

input_size = 2
sequence_length = 10
num_layers = 1


# load data
df = pd.read_csv('data/results_thompson.csv')

choices = df['Action'].to_numpy().reshape(30, 20, 10)
rewards = df['Reward'].to_numpy().reshape(30, 20, 10)
data = np.stack((choices, rewards), axis=-1)  # Shape: (30, 20, 10, 2)

# Reshape to (600, 10, 2) and transpose to get (10, 600, 2) for RNN
data = data.reshape(-1, 10, 2)  # Shape: (600, 10, 2)
data = np.transpose(data, (1, 0, 2))  # Shape: (10, 600, 2)

data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

# Define the split index
split_idx = int(0.8 * data_tensor.shape[1])  # 80% for training

# Split data into training and testing sets
train_tensor = data_tensor[:, :split_idx, :]  # Shape: (10, 480, 2)
test_tensor = data_tensor[:, split_idx:, :]   # Shape: (10, 120, 2)

# Prepare labels for next choice prediction
# Shift the choice data by one step for each sequence in train_tensor
train_choices = train_tensor[:, :, 0]  # Assuming the choice is the first feature
train_labels = train_choices[1:, :]    # Remove the first step for labels
train_inputs = train_tensor[:-1, :, :]  # Remove the last step for inputs to align


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # x -> (batch_size, sequence_length, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward
        out, _ = self.rnn(x, h0)
        # out: batch_size, seq_length, hidden_size
        # out (N, 28, 128)
        out = out[:, -1, :]
        # out (N, 128)
        # out: batch_size, hidden_size
        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
n_total_steps = train_tensor.shape[1]
# training loop
for epoch in range(num_epochs):
    for i in range(n_total_steps):
        # Get the current input sequence - shape: [sequence_length - 1, input_size]
        inputs = train_inputs[:, i, :].to(device)  # Shape: (9, 2)
        labels = train_labels[:, i].long().to(device)  # Shape: (9,)

        # Reshape inputs to match RNN input: (sequence_length - 1, batch_size, input_size)
        inputs = inputs.reshape(sequence_length - 1, 1, input_size)  # Shape: (9, 1, 2)

        # Forward pass
        outputs = model(inputs).squeeze()  # Shape: (9, 1) → (9,)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# test

# Prepare labels for next choice prediction
test_choices = test_tensor[:, :, 0]    # Extract the choice feature
test_labels = test_choices[1:, :]      # Shift by one for the labels
test_inputs = test_tensor[:-1, :, :]   # Shift inputs to align with labels
probs = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i in range(test_inputs.shape[1]):
        inputs = test_inputs[:, i, :].to(device)  # Shape: (sequence_length - 1, input_size)
        labels = test_labels[:, i].long().to(device)  # Shape: (sequence_length - 1,)
        # Reshape for RNN input
        inputs = inputs.reshape(sequence_length - 1, 1, input_size)  # Shape: (9, 1, 2)

        # Forward pass
        outputs = model(inputs).squeeze()  # Shape: (sequence_length - 1,) -> (9,)
        probabilities = torch.softmax(outputs, dim=-1)  # Shape: (sequence_length, batch_size, 2)
        probs.append(probabilities)

        # value, index
        # Predict choices
        _, predictions = torch.max(outputs, 1) if outputs.ndim > 1 else (
                    outputs > 0.5).int()  # For binary classification        n_samples += labels.shape[0]
        print(f'Predictions: {predictions}')
        # Calculate accuracy
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc:.2f}%')
    result_probs = torch.cat(probs, dim=0)  # Shape: (sequence_length, batch_size)
    p_a0 = result_probs[:, 0].cpu().numpy()
    df_rnn_thompson = pd.DataFrame(p_a0)
    df_rnn_thompson.to_csv('data/rnn_thompson.csv', index=False)
    print(result_probs[:,0].shape)
