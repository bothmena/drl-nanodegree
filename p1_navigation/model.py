from torch import nn


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers_units: list):
        super(DQN, self).__init__()

        layers = [
            nn.Linear(state_size, hidden_layers_units[0]),
        ]
        for i, units in enumerate(hidden_layers_units):
            next_units = hidden_layers_units[i+1] if i < len(hidden_layers_units) - 1 else action_size
            layers.append(nn.ReLU())
            layers.append(nn.Linear(units, next_units))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.fc_layers(state)
