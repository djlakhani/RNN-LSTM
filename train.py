from util import *
from train import *
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np

from tqdm import tqdm

from shakespeare_lstm import LSTMModel
from shakespeare_rnn import RNNModel

def train(model, device, train_dataloader, val_dataloader, config):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    epochs = config['epochs']
    patience = config['patience']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            train_loss = train_loss + loss.item()
            batch_count = batch_count + 1

            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = eval(model, device, val_dataloader)
        train_loss = train_loss / batch_count
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        print(f"Epoch: {epoch}, Validation Loss: {val_loss}, Train Loss: {train_loss} ")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model_lstm_tf_200.pth')
        else:
            patience_counter = patience_counter + 1
            if patience_counter > patience:
                print(f"Early stopping triggered at {epoch}!")
                break

    return train_losses, val_losses
        

def eval(model, device, val_dataloader):
    val_losses = []
    criterion = nn.CrossEntropyLoss()
    model.eval()

    for i, (inputs, labels) in enumerate(val_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    
    return val_loss
