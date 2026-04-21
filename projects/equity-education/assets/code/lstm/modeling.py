import copy
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2,
                 lstm_dropout=0.2, output_dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(output_dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.best_state = None
        self.stopped = False
        self.stop_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
            self.stop_epoch = epoch
            return True
        return False

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
