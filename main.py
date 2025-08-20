import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import random_split

from util import encode_text, create_sequences, plot_losses
from shakespeare_dataset import ShakespeareDataset

from shakespeare_lstm import LSTMModel
from shakespeare_rnn import RNNModel
from config import load_config
from train import train, eval

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config_file_path',
        type=str
    )

    args = parser.parse_args()

    input_file_path = 'data/tiny_shakespeare.txt'

    print('ENCODED TEXT DATA')

    encoded_text, vocab_size, char_to_idx, idx_to_char = encode_text(input_file_path)

    config = load_config(args.config_file_path)

    seq_length = config['seq_len']
    X, y = create_sequences(encoded_text, seq_length)

    print('CREATED SEQUENCES')

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)

    len_data = len(y_tensor)

    train_frac, val_frac, test_frac = 0.8, 0.1, 0.1  
    train_size = int(train_frac * len_data)
    val_size = int(val_frac * len_data)
    test_size = len_data - train_size - val_size  

    torch.manual_seed(0)
    indices = torch.randperm(len_data)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    assert set(train_indices).isdisjoint(set(val_indices)) and set(train_indices).isdisjoint(set(test_indices)) and set(val_indices).isdisjoint(set(test_indices))
    print('PERFORMED TRAIN/VAL/TEST SPLIT')

    # Index tensors to get non-overlapping splits
    # Data will be of the form of sequences of indices
    # Each index corresponds to a character
    X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
    X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
    X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


    # Creating dataloaders
    train_dataset = ShakespeareDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = ShakespeareDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = ShakespeareDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(vocab_size=vocab_size, hidden_size=config['hidden_size'],
                                embed_size=config['embed_size'],
                                num_layers=config['num_layers']).to(device)

    config = load_config(args.config_file_path)

    ## TRAIN - will save best model weights
    train_losses, val_losses = train(model=model,
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config)

    ## INFERENCE
    test_loss = eval(model=model, device=device,
    val_dataloader=test_dataloader)

    print(f"Final test loss: {test_loss}")

    plot_losses(train_losses, val_losses, 'lstm_tf_200')


if __name__ == '__main__':
    main()