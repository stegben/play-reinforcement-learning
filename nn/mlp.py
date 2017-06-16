import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_structure, output_dim):
        super(MLP, self).__init__()
        all_layers_dim = [input_dim] + hidden_structure + [output_dim]
        layers = []

        for idx in range(len(all_layers_dim) - 1):
            n_in = all_layers_dim[idx]
            n_out = all_layers_dim[idx + 1]

            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.model.forward(x))
