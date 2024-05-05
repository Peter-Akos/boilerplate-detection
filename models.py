import torch.nn as nn
from torch.nn.functional import relu, sigmoid
import torch


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        res = self.fc1(out)
        res = relu(res)
        return sigmoid(self.fc2(res))


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # Multiply by 2 due to bidirectional LSTM
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply num_layers by 2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        res = self.fc1(out)
        res = relu(res)
        return sigmoid(self.fc2(res))


class LSTMModelV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModelV2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = relu(self.fc1(x))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        return sigmoid(self.fc2(out))


class BiLSTMModelV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLSTMModelV2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(input_size, hidden_size)  # Multiply by 2 due to bidirectional LSTM
        self.fc2 = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = relu(self.fc1(x))
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply num_layers by 2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        return sigmoid(self.fc2(out))


class BiLSTMModelV3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(BiLSTMModelV3, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(input_size, hidden_size)  # Multiply by 2 due to bidirectional LSTM
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = relu(self.fc1(x))
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply num_layers by 2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = relu(out)
        out = self.fc2(out)
        out = relu(out)
        return sigmoid(self.fc3(out))


class EmbeddingBagModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, output_size: int, hidden_size: int, dropout) -> None:
        super(EmbeddingBagModel, self).__init__()
        self.embedding_bag = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.dense1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int)
        x = self.embedding_bag(x).to(torch.float)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return sigmoid(x)


class EmbeddingBagModelV2(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, output_size: int, hidden_size: int, dropout) -> None:
        super(EmbeddingBagModelV2, self).__init__()
        self.embedding_bag = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.dense1 = nn.Linear(64 + embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = x[:, :64]
        x_2 = x[:, 64:].to(torch.int)
        x_2 = self.embedding_bag(x_2).to(torch.float)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return sigmoid(x)

