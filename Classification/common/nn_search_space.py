"""Default hyperparameter search spaces for neural network optimization."""

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
    {"name": "activation_3", "bounds": ["sigmoid"], "type": "cat"},
    {"name": "kernel_init", "bounds": [0, 9], "type": "int"},
    {"name": "bias_init", "bounds": [0, 9], "type": "int"},
    {"name": "kernel_reg", "bounds": ["L1", "L2", "L1L2"], "type": "cat"},
    {"name": "bias_reg", "bounds": ["L1", "L2", "L1L2"], "type": "cat"},
    {"name": "activity_reg", "bounds": ["L1", "L2", "L1L2"], "type": "cat"},
    {
        "name": "kernel_const",
        "bounds": ["MaxNorm", "MinMaxNorm", "NonNeg", "UnitNorm"],
        "type": "cat",
    },
    {
        "name": "bias_const",
        "bounds": ["MaxNorm", "MinMaxNorm", "NonNeg", "UnitNorm"],
        "type": "cat",
    },
    {
        "name": "optim",
        "bounds": [
            "Adadelta",
            "Adagrad",
            "Adam",
            "Adamax",
            "Ftrl",
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
