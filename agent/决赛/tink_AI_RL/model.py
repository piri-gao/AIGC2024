import torch
import torch.nn as nn

class MultiAgentTransformer(nn.Module):
    def __init__(self, n_agents, input_dim, hidden_dim, output_dim, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers, dropout=dropout)

        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        # inputs: (batch_size, n_agents, input_dim)
        batch_size, n_agents, input_dim = inputs.size()

        # Reshape inputs to be (batch_size * n_agents, input_dim)
        inputs = inputs.view(batch_size * n_agents, input_dim)

        # Embed inputs
        embedded = self.embedding(inputs)

        # Reshape embedded to be (n_agents, batch_size, hidden_dim)
        embedded = embedded.view(batch_size, n_agents, -1).transpose(0, 1)

        # Use transformer to process inputs
        outputs = self.transformer(embedded, embedded)

        # Reshape outputs to be (batch_size * n_agents, hidden_dim)
        outputs = outputs.transpose(0, 1).contiguous().view(batch_size * n_agents, -1)

        # Use output layer to generate output vectors
        outputs = self.output(outputs)

        # Reshape outputs to be (batch_size, n_agents, output_dim)
        outputs = outputs.view(batch_size, n_agents, -1)

        return outputs