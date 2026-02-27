import torch.nn as nn


class NeuralNet(nn.Module):
    """
    Enhanced 4-layer MLP with Dropout for better generalization.
    hidden_size is now much larger to capture the expanded intent vocabulary.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)
