import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
from torch import nn, optim
import torch

import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

from utils import CustomDataset, get_split
from models import LSTMModel, BiLSTMModel, LSTMModelV2, BiLSTMModelV2, BiLSTMModelV3, EmbeddingBagModel, \
    EmbeddingBagModelV2
from sweep_config import sweep_config
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DATA_PATH = "cleaneval.csv"
SPLIT = "80-20"
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 2
DROPOUT = 0.5

RUN_NR = 1

TRAINING_TYPE = "SWEEP"
# TRAINING_TYPE = "RUN"
# TRAINING_TYPE = "AGENT"
SWEEP_ID = "clu9lkq1"
SAVE_PATH = ""


def configure_wandb():
    global SAVE_PATH
    if TRAINING_TYPE == "RUN":
        wandb.init(
            project="boilerplate-detection",
            config={
                "learning_rate": LEARNING_RATE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dataset": f"{DATA_PATH.split('.')[0]}-{SPLIT}",
                "epochs": NUM_EPOCHS,
                "dropout": DROPOUT
            }
        )
        train()
    elif TRAINING_TYPE == "SWEEP":
        sweep_id = wandb.sweep(sweep_config, project="boilerplate-detection")
        global SWEEP_ID
        SWEEP_ID = sweep_id
        SAVE_PATH = f"models/{SWEEP_ID}/agent-{datetime.now().strftime('%H:%M:%S')}"
        os.makedirs(SAVE_PATH)
        wandb.agent(sweep_id, train, count=50)
    else:
        SAVE_PATH = f"models/{SWEEP_ID}/agent-{datetime.now().strftime('%H:%M:%S')}"
        os.makedirs(SAVE_PATH)
        wandb.agent(SWEEP_ID, train)


def get_X_and_y(current_ids, df, input_subset):
    curr_X = []
    curr_y = []
    for current_id in current_ids:
        current_id = int(current_id)
        if input_subset == "all":
            filtered_rows = df[df['0'] == current_id].iloc[:, 2:642].to_numpy()
        elif input_subset == "html_only":
            filtered_rows = df[df['0'] == current_id].iloc[:, 2:66].to_numpy()
        elif input_subset == "graph_only":
            filtered_rows = df[df['0'] == current_id].iloc[:, 66:130].to_numpy()
        elif input_subset == "text_only":
            filtered_rows = df[df['0'] == current_id].iloc[:, 130:642].to_numpy()
        elif input_subset == "html+graph":
            filtered_rows = df[df['0'] == current_id].iloc[:, 2:130].to_numpy()
        elif input_subset == "html+text":
            filtered_rows = df[df['0'] == current_id].iloc[:, np.r_[2:66, 130:642]].to_numpy()
        elif input_subset == "graph+text":
            filtered_rows = df[df['0'] == current_id].iloc[:, 66:642].to_numpy()
        elif input_subset == "embeddingbag":
            filtered_rows = df[df['0'] == current_id].iloc[:, 643:].to_numpy()
        elif input_subset == "html+embeddingbag":
            filtered_rows = df[df['0'] == current_id].iloc[:, np.r_[2:66, 643:707]].to_numpy()
        labels = df[df['0'] == current_id].iloc[:, 642].to_numpy()
        curr_X.append(filtered_rows)
        curr_y.append(labels)
    return curr_X, curr_y


def scale(df, scaler, split):
    if scaler is None:
        return
    train_values = df[df.iloc[:, 0].isin([int(x) for x in split[0]])].to_numpy()[:, 2:66]
    validation_values = df[df.iloc[:, 0].isin([int(x) for x in split[1]])].to_numpy()[:, 2:66]
    test_values = df[df.iloc[:, 0].isin([int(x) for x in split[2]])].to_numpy()[:, 2:66]

    scaler.fit(train_values)

    train_df = scaler.transform(train_values)
    validation_df = scaler.transform(validation_values)
    test_df = scaler.transform(test_values)

    scaled_html_values = np.concatenate([train_df, validation_df, test_df])

    df.iloc[:, 2:66] = scaled_html_values


def train_validation_test_split(df, split, input_subset, scaler_name):
    X = []
    y = []

    scaler = None
    if scaler_name == "Standard":
        scaler = StandardScaler()
    elif scaler_name == "MinMax":
        scaler = MinMaxScaler()
    scale(df, scaler, split)

    for current_ids in split:
        curr_X, curr_y = get_X_and_y(current_ids, df, input_subset)
        X.append(curr_X)
        y.append(curr_y)
    return X[0], y[0], X[1], y[1], X[2], y[2]


def load_dataset(split=None, data_path=None, input_subset=None, scaler=None):
    if split is None:
        split = SPLIT
    if data_path is None:
        data_path = DATA_PATH

    df = pd.read_csv(data_path, dtype=float)
    document_ids = df['0'].unique()
    X = []
    y = []

    if split == "80-20":
        for i, val in enumerate(document_ids):
            # 0: document nr, 1: block nr 2:
            filtered_rows = df[df['0'] == val].iloc[:, 3:].to_numpy()
            labels = df[df['0'] == val].iloc[:, 642].to_numpy()
            X.append(filtered_rows)
            y.append(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_validation = X_test
        y_validation = y_test

    elif split == "55-5-676":
        split = get_split()
        X_train, y_train, X_validation, y_validation, X_test, y_test = train_validation_test_split(df, split,
                                                                                                   input_subset, scaler)

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset)
    validation_dataset = CustomDataset(X_validation, y_validation)
    validation_loader = DataLoader(validation_dataset)
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset)
    input_size = X_train[0].shape[1]
    return train_loader, validation_loader, test_loader, input_size


def log_metrics(model, test_loader, criterion, split, save_table=False, ret_cls_report=False):
    model.eval()
    predicted_labels = []
    true_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if len(inputs.shape) > 2:
                inputs = inputs.squeeze(0)
            outputs = model(inputs).T

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            predicted = torch.round(outputs)
            to_extend_with = predicted.squeeze().tolist()
            if isinstance(to_extend_with, float):
                predicted_labels.append(to_extend_with)
            else:
                predicted_labels.extend(to_extend_with)
            to_extend_with = labels.squeeze().tolist()
            if isinstance(to_extend_with, float):
                true_labels.append(to_extend_with)
            else:
                true_labels.extend(to_extend_with)
        test_loss /= len(test_loader.dataset)

    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    metrics = classification_report(true_labels, predicted_labels, output_dict=True)
    f1 = float(metrics["macro avg"]["f1-score"])
    accuracy = metrics.pop("accuracy")

    res = {f"{split}_accuracy": accuracy, f"{split}_loss": test_loss, f"{split}_f1": f1}

    if save_table:
        table_to_log = pd.DataFrame(metrics).T
        t = wandb.Table(dataframe=table_to_log)
        wandb.log({f"Classification report on test for f1: {f1}": t})

    if ret_cls_report:
        return res, metrics

    return res

    # global BEST_VALIDATION_F1
    # if split == "validation" and float(f1) > BEST_VALIDATION_F1:
    #     BEST_TEST_F1 = float(f1)
    #     table_to_log = pd.DataFrame(metrics).T
    #     t = wandb.Table(dataframe=table_to_log)
    #     wandb.log({f"Best f1: {f1}": t})


def build_model(model_nr, hidden_size, num_layers, dropout, input_size):
    if model_nr == 1:
        return LSTMModel(input_size, hidden_size, num_layers, 1, dropout)
    elif model_nr == 2:
        return BiLSTMModel(input_size, hidden_size, num_layers, 1, dropout)
    elif model_nr == 3:
        return LSTMModelV2(input_size, hidden_size, num_layers, 1, dropout)
    elif model_nr == 4:
        return BiLSTMModelV2(input_size, hidden_size, num_layers, 1, dropout)
    elif model_nr == 5:
        return BiLSTMModelV3(input_size, hidden_size, num_layers, 1, dropout)
    elif model_nr == 6:
        return EmbeddingBagModel(hidden_size=hidden_size, embedding_dim=input_size, vocab_size=100000, output_size=1,
                                 dropout=dropout)
    elif model_nr == 7:
        return EmbeddingBagModelV2(hidden_size=hidden_size, embedding_dim=input_size, vocab_size=100000, output_size=1,
                                   dropout=dropout)


def train(config=None):
    if TRAINING_TYPE == "RUN":  # we are in a run
        print("Starting run")
        model = LSTMModel(640, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT)
        run_nr = 0
        train_loader, validation_loader, test_loader = load_dataset()
        num_epochs = NUM_EPOCHS
        lr = LEARNING_RATE

    else:  # we are in a sweep
        print("Starting run in a sweep")
        run = wandb.init(config=config)
        global RUN_NR
        run_nr = RUN_NR
        RUN_NR += 1
        config = wandb.config
        train_loader, validation_loader, test_loader, input_size = load_dataset(config.split, config.data_path,
                                                                                config.input_subset, config.scaler)
        model = build_model(config.model_nr, config.hidden_size, config.num_layers, config.dropout, input_size)
        num_epochs = config.num_epochs
        lr = config.learning_rate

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("train_accuracy", summary="max")
    wandb.define_metric("train_f1", summary="max")

    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("test_accuracy", summary="max")
    wandb.define_metric("test_f1", summary="max")

    wandb.define_metric("validation_loss", summary="min")
    wandb.define_metric("validation_accuracy", summary="max")
    # wandb.define_metric("validation_f1", summary="max", goal="minimize")

    print("Starting Training")
    best_validation_f1 = 0
    models_nr = 0
    results = []  # run_nr, model_nr, validation_f1, test_f1
    best_state_dict = {}

    for _ in tqdm(range(num_epochs)):
        for inputs, labels in train_loader:
            inputs = inputs.squeeze()
            optimizer.zero_grad()
            outputs = model(inputs).T
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # log_metrics(model, train_loader, criterion, "train")
        metrics = log_metrics(model, validation_loader, criterion, "validation")
        f1_vali = metrics['validation_f1']
        if f1_vali > best_validation_f1:
            best_validation_f1 = f1_vali
            best_state_dict = copy.deepcopy(model.state_dict())

        wandb.log(metrics)
        metrics = log_metrics(model, test_loader, criterion, "test")
        wandb.log(metrics)

        t = wandb.Table(data=results, columns=["Run number", "Model number", "Validation f1-score", "Test f1-score"])
        wandb.log({f"Best models": t})

    print("Finished Training")
    print(f"Loading model with validation f1 score of {best_validation_f1}")
    model.load_state_dict(best_state_dict)
    test_metrics = log_metrics(model, test_loader, criterion, "test")
    f1_test = test_metrics['test_f1']
    print(f"f1-score of the best model on test: {f1_test}")
    results.append([run_nr, best_validation_f1, f1_test])
    wandb.log(test_metrics)
    torch.save(model.state_dict(),
               f'{SAVE_PATH}/{run_nr}.pth')

    if TRAINING_TYPE == "SWEEP":
        run.finish()


if __name__ == "__main__":
    configure_wandb()
