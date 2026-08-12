"""Shared Classification utilities."""

from common.data import DataProcessor
from common.features import Scaling_and_FeatureExtractor
from common.helpers import most_common, warn
from common.nn_hyperparameter_search import GaHyperparamSearcher, HyperparamSearcher, PsoHyperparamSearcher
from common.nn_model import BaselineNnBuilder
from common.nn_search_space import (
    DEFAULT_GA_ALGORITHM_PARAMETERS,
    DEFAULT_GA_OBJECTIVE_PARAMETERS,
    DEFAULT_PSO_BOUNDS_MAX,
    DEFAULT_PSO_BOUNDS_MIN,
    DEFAULT_PSO_NAME_HYPERPARAM,
)
from common.nn_train import DEFAULT_DEVICE, predict_classes, predict_proba, train_model
from common.plotting import SpectralPlotter

__all__ = [
    "most_common",
    "warn",
    "DataProcessor",
    "Scaling_and_FeatureExtractor",
    "SpectralPlotter",
    "BaselineNnBuilder",
    "HyperparamSearcher",
    "GaHyperparamSearcher",
    "PsoHyperparamSearcher",
    "DEFAULT_GA_OBJECTIVE_PARAMETERS",
    "DEFAULT_GA_ALGORITHM_PARAMETERS",
    "DEFAULT_PSO_NAME_HYPERPARAM",
    "DEFAULT_PSO_BOUNDS_MIN",
    "DEFAULT_PSO_BOUNDS_MAX",
    "DEFAULT_DEVICE",
    "train_model",
    "predict_proba",
    "predict_classes",
]
