"""Default hyperparameter search spaces for neural network optimization."""

import numpy as np

DEFAULT_GA_OBJECTIVE_PARAMETERS = [
    {"name": "neuron_layer_1", "bounds": [10, 15], "type": "int"},
    {"name": "neuron_layer_2", "bounds": [5, 9], "type": "int"},
    {
        "name": "activation_1",
        "bounds": [
            "relu",
            "softmax",
            "sigmoid",
            "softplus",
            "softsign",
            "tanh",
            "selu",
            "elu",
            "exponential",
        ],
        "type": "cat",
    },
    {
        "name": "activation_2",
        "bounds": [
            "relu",
            "softmax",
            "sigmoid",
            "softplus",
            "softsign",
            "tanh",
            "selu",
            "elu",
            "exponential",
        ],
        "type": "cat",
    },
    {"name": "kernel_init", "bounds": [0, 9], "type": "int"},
    {"name": "bias_init", "bounds": [0, 9], "type": "int"},
    {"name": "kernel_reg", "bounds": ["L1", "L2", "L1L2"], "type": "cat"},
    {"name": "bias_reg", "bounds": ["L1", "L2", "L1L2"], "type": "cat"},
    {
        "name": "optim",
        "bounds": [
            "Adadelta",
            "Adagrad",
            "Adam",
            "Adamax",
            "Nadam",
            "RMSprop",
            "SGD",
        ],
        "type": "cat",
    },
]

DEFAULT_GA_ALGORITHM_PARAMETERS = {
    "mutation_probability": 0.1,
    "elite_ratio": 0.05,
    "crossover_probability": 0.5,
    "parents_portion": 0.3,
    "crossover_type": "uniform",
    "max_iteration_without_improv": None,
}

# PSO bounds aligned with DEFAULT_GA_OBJECTIVE_PARAMETERS (9 dimensions).
DEFAULT_PSO_NAME_HYPERPARAM = [
    "neuron_layer_1",
    "neuron_layer_2",
    "activation_1",
    "activation_2",
    "kernel_init",
    "bias_init",
    "kernel_reg",
    "bias_reg",
    "optim",
]

DEFAULT_PSO_BOUNDS_MAX = np.array([15, 9, 8, 8, 9, 9, 2, 2, 6])
DEFAULT_PSO_BOUNDS_MIN = np.array([10, 5, 0, 0, 0, 0, 0, 0, 0])
