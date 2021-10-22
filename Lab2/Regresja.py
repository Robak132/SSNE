import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

### Spróbujmy przewidzieć ocenę wina na podstawie jego parametrów


class WineDataset(data.Dataset):
    def __init__(self, _data):
        super().__init__()
        self.data = _data[:, :-1]
        # self.data = self.data / self.data.max(axis=0)
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
    def __init__(self, num_inputs, num_hidden1, num_outputs, device):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_outputs)
        self.act_fn = torch.nn.modules.activation.ReLU()
        self.to(device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


def train_network(max_epoch):
    train_dataset = WineDataset(train.to_numpy())
    train_data_loader = data.DataLoader(train_dataset, batch_size=50, shuffle=True, drop_last=True)

    model.train()
    # Training loop
    for epoch in range(max_epoch):
        for data_inputs, data_labels in train_data_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs.float())
            preds = preds.squeeze(dim=1)  # Output is [Batch size, 1], but we want [Batch size]

            loss = loss_module(preds, data_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0 or epoch == max_epoch-1:
            print(f"Epoch = {epoch}, loss = {loss.item()}")


def test_network():
    test_dataset = WineDataset(test.to_numpy())
    test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():  # Deactivate gradients for the following code
        error = 0
        good = 0
        for data_inputs, data_labels in test_data_loader:
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            preds = model(data_inputs.float())
            preds = preds.squeeze(dim=1)

            print(f'Predicted: {preds.item():0.4f} [{round(preds.item())}], Real: {int(data_labels.item())}')
            if round(preds.item()) == int(data_labels.item()):
                good += 1
            error += (data_labels - preds).abs().item()
    error = error / len(test_dataset)
    good = good / len(test_dataset)
    print()
    print(f"Average error: {error:.5f}")
    print(f"Integer mode efficiency: {good * 100:.2f}%")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not enabled")
        quit(-1)

    device = torch.device("cuda")
    torch.cuda.manual_seed_all(2137)
    torch.manual_seed(2137)

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', delimiter=";")
    train = df.sample(frac=0.9, random_state=2137)
    test = df.drop(train.index)

    model = SimpleEstimator(11, 15, 1, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_module = nn.L1Loss()

    train_network(1000)
    test_network()

