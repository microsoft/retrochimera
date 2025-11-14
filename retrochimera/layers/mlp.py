from torch import nn


class MLP(nn.Module):
    """Simple MLP with at least two layers and all hidden layers of the same size."""

    def __init__(
        self, in_dim, hidden_dim, out_dim, n_layers=1, activation=nn.ELU, dropout=0.2
    ) -> None:
        super(MLP, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.Dropout(dropout), activation()]

        for _ in range(n_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), activation()]

        layers += [nn.Linear(hidden_dim, out_dim)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
