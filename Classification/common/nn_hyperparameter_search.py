"""Hyperparameter searcher classes for neural network optimization."""

import numpy as np
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.metrics import balanced_accuracy_score

from common.nn_model import BaselineNnBuilder
from common.nn_search_space import (
    DEFAULT_GA_ALGORITHM_PARAMETERS,
    DEFAULT_GA_OBJECTIVE_PARAMETERS,
    DEFAULT_PSO_BOUNDS_MAX,
    DEFAULT_PSO_BOUNDS_MIN,
    DEFAULT_PSO_NAME_HYPERPARAM,
)
from common.nn_train import DEFAULT_DEVICE, predict_classes, train_model

PSO_ACTIVATIONS = DEFAULT_GA_OBJECTIVE_PARAMETERS[2]["bounds"]
PSO_REGULATION = DEFAULT_GA_OBJECTIVE_PARAMETERS[6]["bounds"]
PSO_OPTIMIZERS = DEFAULT_GA_OBJECTIVE_PARAMETERS[8]["bounds"]


def decode_pso_position(pos):
    """Map a PSO particle vector to BaselineNnBuilder hyperparameters."""
    return {
        "neuron_layer_1": int(np.round(pos[0])),
        "neuron_layer_2": int(np.round(pos[1])),
        "activation_1": PSO_ACTIVATIONS[int(np.round(pos[2]))],
        "activation_2": PSO_ACTIVATIONS[int(np.round(pos[3]))],
        "kernel_init": int(np.round(pos[4])),
        "bias_init": int(np.round(pos[5])),
        "kernel_reg": PSO_REGULATION[int(np.round(pos[6]))],
        "bias_reg": PSO_REGULATION[int(np.round(pos[7]))],
        "optim": PSO_OPTIMIZERS[int(np.round(pos[8]))],
    }


class HyperparamSearcher:
    """Shared training, scoring, and result helpers for NN hyperparameter search."""

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=200,
        batch_size=32,
        model_builder=None,
        device=DEFAULT_DEVICE,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_builder = model_builder or BaselineNnBuilder()
        self.device = device
        self.input_dim = X_train.shape[1]

        self.scores_train = []
        self.scores_val = []

    def _train_and_score(self, hyperparams):
        """Fit a model and return train/validation balanced accuracy scores."""
        model = self.model_builder.build(self.input_dim, **hyperparams)
        train_model(
            model,
            self.X_train,
            self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            kernel_reg=hyperparams.get("kernel_reg"),
            bias_reg=hyperparams.get("bias_reg"),
            optim=hyperparams.get("optim", "Adam"),
            device=self.device,
        )

        preds_train = predict_classes(model, self.X_train, device=self.device)
        preds_val = predict_classes(model, self.X_val, device=self.device)

        score_train = balanced_accuracy_score(self.y_train, preds_train)
        score_val = balanced_accuracy_score(self.y_val, preds_val)

        print("DNN")
        print(f"Score on training set:   balanced_accuracy={score_train * 100:.1f}%")
        print(f"Score on validation set: balanced_accuracy={score_val * 100:.1f}%")
        print(" ")
        print(" ")

        self.scores_train.append(score_train)
        self.scores_val.append(score_val)

        return score_train, score_val

    def best_fold_scores(self):
        """Return train and validation scores for the best validation candidate."""
        best_val = max(self.scores_val)
        best_train = self.scores_train[self.scores_val.index(best_val)]
        return best_train, best_val


class GaHyperparamSearcher(HyperparamSearcher):
    """Run GA hyperparameter search for a baseline Dense neural network."""

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=200,
        batch_size=32,
        function_timeout=300,
        objective_parameters=None,
        algorithm_parameters=None,
        model_builder=None,
        device=DEFAULT_DEVICE,
    ):
        super().__init__(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_builder=model_builder,
            device=device,
        )
        self.function_timeout = function_timeout
        self.objective_parameters = objective_parameters or DEFAULT_GA_OBJECTIVE_PARAMETERS
        self.algorithm_parameters = algorithm_parameters or DEFAULT_GA_ALGORITHM_PARAMETERS.copy()
        self.evo_algo = None

    def evaluate(self, args):
        """Train a candidate model and return negative validation balanced accuracy."""
        _, score_val = self._train_and_score(args)
        return score_val * -1

    def run(self, max_num_iteration=10, population_size=200):
        """Run the evolutionary algorithm over the configured search space."""
        algorithm_parameters = self.algorithm_parameters.copy()
        algorithm_parameters["max_num_iteration"] = max_num_iteration
        algorithm_parameters["population_size"] = population_size

        self.evo_algo = ea(
            function=self.evaluate,
            parameters=self.objective_parameters,
            function_timeout=self.function_timeout,
            algorithm_parameters=algorithm_parameters,
        )
        self.evo_algo.run()

    def best_parameter_names(self):
        """Return ordered hyperparameter names from the best GA candidate."""
        return list(self.evo_algo.best_parameters.keys())

    def best_parameter_values(self):
        """Return ordered hyperparameter values from the best GA candidate."""
        return list(self.evo_algo.best_parameters.values())


class PsoHyperparamSearcher(HyperparamSearcher):
    """Run PSO hyperparameter search for a baseline Dense neural network."""

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=200,
        batch_size=32,
        max_num_iteration=2,
        n_particles=5,
        model_builder=None,
        device=DEFAULT_DEVICE,
    ):
        super().__init__(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_builder=model_builder,
            device=device,
        )
        self.max_num_iteration = max_num_iteration
        self.n_particles = n_particles
        self.cost = None
        self.pos = None

    def objective_function(self, particles):
        """Evaluate all particles; pyswarms expects costs to be minimized."""
        costs = []
        for particle in particles:
            hyperparams = decode_pso_position(particle)
            _, score_val = self._train_and_score(hyperparams)
            costs.append(score_val * -1)
        return np.array(costs)

    def run(self):
        """Run GlobalBestPSO over the configured search space."""
        from pyswarms.single.global_best import GlobalBestPSO

        bounds = (DEFAULT_PSO_BOUNDS_MIN, DEFAULT_PSO_BOUNDS_MAX)
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(DEFAULT_PSO_NAME_HYPERPARAM),
            options=options,
            bounds=bounds,
        )
        self.cost, self.pos = optimizer.optimize(
            self.objective_function,
            iters=self.max_num_iteration,
            verbose=False,
        )

    def best_parameter_names(self):
        """Return ordered hyperparameter names from the best PSO candidate."""
        return DEFAULT_PSO_NAME_HYPERPARAM.copy()

    def best_parameter_values(self):
        """Return ordered hyperparameter values from the best PSO candidate."""
        decoded = decode_pso_position(self.pos)
        return [decoded[name] for name in DEFAULT_PSO_NAME_HYPERPARAM]
