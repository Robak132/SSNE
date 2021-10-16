import torch
import torch.utils.data as data
import torch.nn as nn

import numpy as np
import time
import math
import pandas as pd


### Spróbujmy przewidzieć ocenę wina na podstawie jego parametrów


class WineDataset(data.Dataset):
    def __init__(self, _data):
        super().__init__()
        self.data = _data[:, :-1]
        self.label = _data[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label


class SimpleEstimator(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, _device):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.to(_device)

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


def train_network():
    train_dataset = WineDataset(train.to_numpy())
    train_data_loader = data.DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(100):
        for data_inputs, data_label in train_data_loader:
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_label = data_label.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs.float())
            preds = preds.squeeze(dim=1)  # Output is [Batch size, 1], but we want [Batch size]

            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_label.float())

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()
        # print(f"Epoch: {epoch}, loss: {loss.item():.3}")


def test_network():
    test_dataset = WineDataset(test.to_numpy())
    test_data_loader = data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    model.eval()
    with torch.no_grad():  # Deactivate gradients for the following code
        for data_inputs, data_label in test_data_loader:
            data_inputs = data_inputs.to(device)
            data_label = data_label.to(device)

            prediction = model(data_inputs.float())
            prediction = prediction.squeeze(dim=1)

            error = (data_label - prediction).abs().mean()
            error2 = ((data_label - prediction).abs() ** 2).mean()
    print(f"Average error: {error:.5f}")
    print(f"Average quadratic error: {error2:.5f}\n")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not enabled")
        quit(-1)

    device = torch.device("cuda")

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', delimiter=";")
    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)

    model = SimpleEstimator(11, 6, 1, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_module = nn.MSELoss()

    train_network()
    test_network()
