import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


class MyDataset(data.Dataset):
    def __init__(self, _data):
        super().__init__()
        self.data_value = _data[:, 1:]
        self.label = np.where(_data[:, 0] > 300000, 1, 0)

    def __len__(self):
        return len(self.data_value)

    def __getitem__(self, idx):
        data_point = self.data_value[idx]
        data_label = self.label[idx]
        return data_point, data_label


class SimpleEstimator(nn.Module):
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


def train_network(set, max_epoch):
    train_dataset = MyDataset(set.to_numpy())
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


def test_network(set):
    test_dataset = MyDataset(set.to_numpy())
    test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():  # Deactivate gradients for the following code
        error = 0
        for data_value, label in test_data_loader:
            data_value = data_value.to(device)

            preds = model(data_value.float())
            preds = preds.squeeze(dim=1)

            error += abs(preds.item() - label.item())
        error = error / len(set)
        print(f"Total error: {error}")


def change_time(string):
    try:
        value = int(string.replace("min", "").replace("~", "-").split("-")[1])
    except IndexError:
        value = 999
    return value


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not enabled")
        quit(-1)

    device = torch.device("cuda")
    torch.cuda.manual_seed_all(2137)
    torch.manual_seed(2137)

    df = pd.read_csv('train_data.csv', delimiter=",")
    df['TimeToBusStop'] = df['TimeToBusStop'].map(lambda a: change_time(a))
    df['TimeToSubway'] = df['TimeToSubway'].map(lambda a: change_time(a))
    df = df.drop(columns=['Floor', 'HallwayType', 'HeatingType', 'AptManageType', 'SubwayStation'])

    train = df.sample(frac=0.8, random_state=2137)
    validation = df.drop(train.index)

    model = SimpleEstimator(24, 16, 8, 1, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_module = nn.L1Loss()

    train_network(train, 750)
    test_network(validation)