import torch
import torch.nn as nn

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size, num_layers):
        super(RNNModel, self).__init__()
        
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embeded = self.embedding_layer(x)        
        output, h_t = self.rnn(embeded)
        return self.linear(output)[:, -1, :]