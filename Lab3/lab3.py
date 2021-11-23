import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


class MyDataset(data.Dataset):
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


class TestDataset(data.Dataset):
    def __init__(self, _data):
        super().__init__()
        self.data = _data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        return data_point


class SimpleEstimator(nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_outputs, device):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_outputs)
        self.act_fn = torch.nn.modules.activation.Sigmoid()
        self.to(device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class SimpleEstimator2(nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs, device):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_outputs)
        self.act_fn = torch.nn.modules.activation.Sigmoid()
        self.to(device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        return x


class SimpleEstimator3(nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_hidden3, num_outputs, device):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_hidden3)
        self.linear4 = nn.Linear(num_hidden3, num_outputs)
        self.act_fn = torch.nn.modules.activation.Sigmoid()
        self.to(device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        x = self.act_fn(x)
        x = self.linear4(x)
        return x


def train_network(max_epoch):
    train_dataset = MyDataset(train.to_numpy())
    train_data_loader = data.DataLoader(train_dataset, batch_size=25, shuffle=True, drop_last=True)

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
    test_dataset = TestDataset(test.to_numpy())
    test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():  # Deactivate gradients for the following code
        error = 0
        with open("robaczewski_pak.csv") as file:
            for data_inputs in test_data_loader:
                data_inputs = data_inputs.to(device)

                preds = model(data_inputs.float())
                preds = preds.squeeze(dim=1)

                file.write(preds.item())
                

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not enabled")
        quit(-1)

    device = torch.device("cuda")
    torch.cuda.manual_seed_all(2137)
    torch.manual_seed(2137)

    train = pd.read_csv('data.csv', delimiter=",")
    train = train.drop(columns=['instant', 'dteday', 'casual', 'registered'])
    test = pd.read_csv('evaluation_data.csv', delimiter=',')

    model = SimpleEstimator2(12, 20, 10, 1, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_module = nn.L1Loss()

    train_network(500)
    test_network()

