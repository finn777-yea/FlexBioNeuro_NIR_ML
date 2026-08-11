from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.metrics import balanced_accuracy_score

from common.nn_model import BaselineNnBuilder
from common.nn_search_space import (
    DEFAULT_GA_ALGORITHM_PARAMETERS,
    DEFAULT_GA_OBJECTIVE_PARAMETERS,
)


class GaHyperparamSearcher:
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
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.function_timeout = function_timeout
        self.objective_parameters = objective_parameters or DEFAULT_GA_OBJECTIVE_PARAMETERS
        self.algorithm_parameters = algorithm_parameters or DEFAULT_GA_ALGORITHM_PARAMETERS.copy()
        self.model_builder = model_builder or BaselineNnBuilder()
        self.input_dim = X_train.shape[1]

        self.scores_train = []
        self.scores_val = []
        self.evo_algo = None

    def evaluate(self, args):
        """Train a candidate model and return negative validation balanced accuracy."""
        model = self.model_builder.build(self.input_dim, **args)
        model.fit(
            x=self.X_train,
            y=self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_data=(self.X_val, self.y_val),
        )

        preds_train = (model.predict(self.X_train) > 0.5).astype("int32")
        preds_val = (model.predict(self.X_val) > 0.5).astype("int32")

        score_train = balanced_accuracy_score(self.y_train, preds_train)
        score_val = balanced_accuracy_score(self.y_val, preds_val)

        print("DNN")
        print(f"Score on training set:   balanced_accuracy={score_train * 100:.1f}%")
        print(f"Score on validation set: balanced_accuracy={score_val * 100:.1f}%")
        print(" ")
        print(" ")

        self.scores_train.append(score_train)
        self.scores_val.append(score_val)

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

    def best_fold_scores(self):
        """Return train and validation scores for the best validation candidate."""
        best_val = max(self.scores_val)
        best_train = self.scores_train[self.scores_val.index(best_val)]
        return best_train, best_val
