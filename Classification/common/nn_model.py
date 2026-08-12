"""PyTorch baseline dense neural network builder."""

import torch
import torch.nn as nn


class _ExpActivation(nn.Module):
    """Element-wise exponential activation (Keras 'exponential' equivalent)."""

    def forward(self, x):
        return torch.exp(x)


def _make_activation(name):
    activations = {
        "relu": nn.ReLU(),
        "softmax": nn.Softmax(dim=-1),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "softsign": nn.Softsign(),
        "tanh": nn.Tanh(),
        "selu": nn.SELU(),
        "elu": nn.ELU(),
        "exponential": _ExpActivation(),
    }
    return activations[name]


# Index order mirrors the former Keras initializer list in BaselineNnBuilder.
INIT_FN_BY_INDEX = {
    0: lambda t: nn.init.normal_(t, mean=0.0, std=0.05),
    1: lambda t: nn.init.uniform_(t, a=-0.05, b=0.05),
    2: lambda t: nn.init.trunc_normal_(t, mean=0.0, std=0.05, a=-0.1, b=0.1),
    3: lambda t: nn.init.xavier_uniform_(t, gain=1.0),
    4: lambda t: nn.init.xavier_normal_(t, gain=1.0),
    5: lambda t: nn.init.xavier_uniform_(t, gain=1.0),
    6: lambda t: nn.init.kaiming_normal_(t, nonlinearity="relu"),
    7: lambda t: nn.init.kaiming_uniform_(t, nonlinearity="relu"),
    8: lambda t: nn.init.normal_(t, mean=0.0, std=(1.0 / t.shape[1]) ** 0.5 if t.dim() > 1 else 0.05),
    9: lambda t: nn.init.uniform_(
        t,
        a=-((1.0 / t.shape[1]) ** 0.5 if t.dim() > 1 else 0.05),
        b=(1.0 / t.shape[1]) ** 0.5 if t.dim() > 1 else 0.05,
    ),
}


class BaselineDenseNet(nn.Module):
    """Three-layer dense binary classifier; final layer outputs logits (no sigmoid)."""

    def __init__(
        self,
        input_dim,
        neuron_layer_1=15,
        neuron_layer_2=8,
        activation_1="elu",
        activation_2="elu",
        kernel_init=7,
        bias_init=5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, neuron_layer_1)
        self.fc2 = nn.Linear(neuron_layer_1, neuron_layer_2)
        # Logits only: sigmoid is applied at prediction time for stable BCEWithLogitsLoss training.
        self.fc3 = nn.Linear(neuron_layer_2, 1)
        self.act1 = _make_activation(activation_1)
        self.act2 = _make_activation(activation_2)

        for layer in (self.fc1, self.fc2, self.fc3):
            INIT_FN_BY_INDEX[kernel_init](layer.weight)
            INIT_FN_BY_INDEX[bias_init](layer.bias)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class BaselineNnBuilder:
    """Build a PyTorch BaselineDenseNet from hyperparameter dicts."""

    def __init__(self, seed=42):
        self.seed = seed

    def build(
        self,
        input_dim,
        neuron_layer_1=15,
        neuron_layer_2=8,
        activation_1="elu",
        activation_2="elu",
        activation_3=None,
        kernel_init=7,
        bias_init=5,
        kernel_reg=None,
        bias_reg=None,
        activity_reg=None,
        kernel_const=None,
        bias_const=None,
        optim="Adam",
    ):
        """Create a BaselineDenseNet; regularization and optimizer are applied during training."""
        torch.manual_seed(self.seed)
        return BaselineDenseNet(
            input_dim=input_dim,
            neuron_layer_1=neuron_layer_1,
            neuron_layer_2=neuron_layer_2,
            activation_1=activation_1,
            activation_2=activation_2,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
