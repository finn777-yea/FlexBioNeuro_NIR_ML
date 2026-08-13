"""PyTorch baseline dense neural network builder."""

import math

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


def _fans(tensor):
    """Fan-in / fan-out; 1D tensors follow Keras (fan_in = fan_out = numel)."""
    if tensor.dim() < 2:
        n = max(tensor.numel(), 1)
        return n, n
    fan_in = tensor.size(1)
    fan_out = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for size in tensor.shape[2:]:
            receptive_field_size *= size
    return fan_in * receptive_field_size, fan_out * receptive_field_size


def _xavier_uniform(tensor, gain=1.0):
    fan_in, fan_out = _fans(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return nn.init.uniform_(tensor, -bound, bound)


def _xavier_normal(tensor, gain=1.0):
    fan_in, fan_out = _fans(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return nn.init.normal_(tensor, 0.0, std)


def _kaiming_normal(tensor, nonlinearity="relu"):
    fan_in, _ = _fans(tensor)
    std = nn.init.calculate_gain(nonlinearity) / math.sqrt(fan_in)
    return nn.init.normal_(tensor, 0.0, std)


def _kaiming_uniform(tensor, nonlinearity="relu"):
    fan_in, _ = _fans(tensor)
    bound = nn.init.calculate_gain(nonlinearity) * math.sqrt(3.0 / fan_in)
    return nn.init.uniform_(tensor, -bound, bound)


def _lecun_normal(tensor):
    fan_in, _ = _fans(tensor)
    return nn.init.normal_(tensor, 0.0, 1.0 / math.sqrt(fan_in))


def _lecun_uniform(tensor):
    fan_in, _ = _fans(tensor)
    bound = math.sqrt(3.0 / fan_in)
    return nn.init.uniform_(tensor, -bound, bound)


# Index order mirrors the former Keras initializer list in BaselineNnBuilder.
# Xavier/Kaiming/Lecun variants must work on 1D biases (Keras did; torch.nn.init does not).
INIT_FN_BY_INDEX = {
    0: lambda t: nn.init.normal_(t, mean=0.0, std=0.05),
    1: lambda t: nn.init.uniform_(t, a=-0.05, b=0.05),
    2: lambda t: nn.init.trunc_normal_(t, mean=0.0, std=0.05, a=-0.1, b=0.1),
    3: lambda t: _xavier_uniform(t, gain=1.0),
    4: lambda t: _xavier_normal(t, gain=1.0),
    5: lambda t: _xavier_uniform(t, gain=1.0),
    6: lambda t: _kaiming_normal(t, nonlinearity="relu"),
    7: lambda t: _kaiming_uniform(t, nonlinearity="relu"),
    8: _lecun_normal,
    9: _lecun_uniform,
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
