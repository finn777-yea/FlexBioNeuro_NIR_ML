# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:45:51 2023

@author: ge23hum
"""


from pathlib import Path
import argparse
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Classification/, for `common`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `paths`

from paths import calibration_path, data_path

from common.helpers import most_common, warn
from common.data import DataProcessor
from common.features import Scaling_and_FeatureExtractor
from common.nn_hyperparameter_search import GaHyperparamSearcher
from common.nn_model import BaselineNnBuilder

import random
import warnings

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

warnings.warn = warn
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

N_SPLITS = 5
SMOKE_EPOCHS = 2
SMOKE_FOLDS = 2
DEFAULT_EPOCHS = 30
GA_MAX_NUM_ITERATION = 10 - 8
GA_POPULATION_SIZE = 1000 - 995


class Tee:
    """Write to multiple streams and flush immediately (terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


def start_tee_log(cal_case):
    """Tee stdout/stderr to a timestamped log under logs/ and return the path and file."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"torch_NN_GA_{cal_case}_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_path, log_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Choose hyperparameters for Torch NN GA (VFA_TA)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to train the model.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pipeline smoke test: 2 epochs, first 2 folds of the 5-fold split.",
    )
    return parser.parse_args()


#%% Main
def process_fold(
        train_ix,
        test_ix,
        X_features_a,
        X_features_b,
        X_features_c,
        y_all,
        groups,
        meas_sec,
        df_all,
        epochs=30,
        batch_size=32,
):
    # Split data for the current fold
    X_train_val_a, X_test_a = X_features_a[train_ix, :],      X_features_a[test_ix, :]
    X_train_val_b, X_test_b = X_features_b[train_ix, :],      X_features_b[test_ix, :]
    X_train_val_c, X_test_c = X_features_c[train_ix, :],      X_features_c[test_ix, :]
    y_train_val,   y_test   = y_all[train_ix], y_all[test_ix]
        
        
        
        
        
    """
    ######################################################################################################
    # Extract features, Class: Scaling_and_FeatureExtractor
    ######################################################################################################
    """
    # Feature Extraction and Predefined Cross-Validation Setup
    features_extractor = Scaling_and_FeatureExtractor(number_of_features = 5, 
                                                      X_train_val_a      = X_train_val_a, 
                                                      X_train_val_b      = X_train_val_b, 
                                                      X_train_val_c      = X_train_val_c, 
                                                      y_train_val        = y_train_val, 
                                                      X_test_a           = X_test_a, 
                                                      X_test_b           = X_test_b, 
                                                      X_test_c           = X_test_c, 
                                                      y_test             = y_test, 
                                                      df_all             = df_all)
        
    features_extractor.Scaling_data()
    features_extractor.extract_features()
    features_extractor.Predefined_cv()

    # Retrieve data after feature extraction and scaling
    X_train=features_extractor.X_out
    y_train=features_extractor.y_train_val
    X_val=features_extractor.X_out_test
    y_val=features_extractor.y_test
        
        
        
        
        
    """
    ######################################################################################################
    # Machine Learning, Class: GaHyperparamSearcher
    ######################################################################################################
    """
    searcher = GaHyperparamSearcher(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        function_timeout=300,
        epochs=epochs,
        batch_size=batch_size,
        model_builder=BaselineNnBuilder(),
    )
    searcher.run(
        max_num_iteration=GA_MAX_NUM_ITERATION,
        population_size=GA_POPULATION_SIZE,
    )
        
        
        
        
        
    """
    ######################################################################################################
    # Save results
    ######################################################################################################
    """
    name_hyperparam, choose_hyperparam = searcher.best_parameter_names(), searcher.best_parameter_values()
    mean_scores_train, mean_scores_val = searcher.best_fold_scores()

    return name_hyperparam, choose_hyperparam, mean_scores_train, mean_scores_val


if __name__ == "__main__":
    """
    ######################################################################################################
    # Calibration case: VFA/TA-ratio or acetic acid concentration
    ######################################################################################################
    """
    args = parse_args()
    cal_case = "VFA_TA"
    if cal_case == "VFA_TA":
        data_file_path = data_path("classification_NIR_Data_raw_VFA_TA.csv")
        calibrate      = "no"#"yes"
    elif cal_case == "Ac_acid":
        data_file_path = data_path("classification_NIR_Data_raw_Ac_acid.csv")
        calibrate      = "yes"

    log_path, log_file = start_tee_log(cal_case)
    epochs = SMOKE_EPOCHS if args.smoke else args.epochs
    batch_size = args.batch_size
    max_folds = SMOKE_FOLDS if args.smoke else N_SPLITS
    if torch.cuda.is_available():
        device_line = "cuda (%s)" % torch.cuda.get_device_name(0)
    else:
        device_line = "CUDA not available"

    print("=" * 72)
    print("Choose hyperparameter: Torch NN GA")
    print("Log file:           " + str(log_path))
    print("Calibration case:   " + cal_case)
    print("Device:             " + device_line)
    print("Smoke:              " + ("yes" if args.smoke else "no"))
    print("Epochs:             " + str(epochs))
    print("Batch size:         " + str(batch_size))
    print("Folds to run:       %d of %d" % (max_folds, N_SPLITS))
    print(
        "GA:                 max_num_iteration=%d, population_size=%d"
        % (GA_MAX_NUM_ITERATION, GA_POPULATION_SIZE)
    )
    print("=" * 72)

    try:
        """
        ######################################################################################################
        # Processing data, Class: DataProcessor
        ######################################################################################################
        """
        # Initialize Data Processor
        data_processor = DataProcessor(data_file_path        = data_file_path,
                                       calibration_file_path = calibration_path(),
                                       meas_sec              = 8)

        # Load raw data, set up features and targets, load calibration data, and perform baseline correction
        data_processor.load_data()
        data_processor.set_XY_groups()
        data_processor.load_calibration(calibrate = calibrate)
        data_processor.randomize()
        X_features_a, _ = data_processor.baseline_correction(method=2)
        X_features_b, _ = data_processor.baseline_correction(method=3)
        X_features_c, _ = data_processor.baseline_correction(method=4)

        """
        ######################################################################################################
        # FOR loop preparation
        ######################################################################################################
        """
        # Initialize Stratified Group K-Fold for cross-validation
        gkf = StratifiedGroupKFold(n_splits=N_SPLITS)

        # Initialize lists to store hyperparameters and mean scores for training and validation sets
        choose_hyperparam = []
        mean_scores_train = list()
        mean_scores_val  = list()
        name_hyperparam = None

        # Sequential fold loop: one GPU-backed training run at a time.
        for fold_idx, (train_ix, test_ix) in enumerate(
            gkf.split(data_processor.X_cal, data_processor.y_all, data_processor.groups),
            start=1,
        ):
            print("")
            print("Fold %d of %d" % (fold_idx, max_folds))
            result1, result2, result3, result4 = process_fold(
                train_ix,
                test_ix,
                X_features_a,
                X_features_b,
                X_features_c,
                data_processor.y_all,
                data_processor.groups,
                data_processor.meas_sec,
                data_processor.df_all,
                epochs=epochs,
                batch_size=batch_size,
            )
            if not len(choose_hyperparam):
                name_hyperparam = result1
                choose_hyperparam = result2
            else:
                choose_hyperparam = np.column_stack((choose_hyperparam, result2))

            mean_scores_train.append(result3)
            mean_scores_val.append(result4)

            if fold_idx >= max_folds:
                break

        # Print results
        print("")
        print("")
        print("Calibration for :  " + cal_case)
        print("Chosen Algorithm:  Deep Neural Network")
        print('mean_scores_train, Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_train)*100, np.std(mean_scores_train)*100))
        print('mean_scores_val,  Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_val)*100, np.std(mean_scores_val)*100))

        print("")
        for i in range(len(name_hyperparam)):
            print("Hyperparameter " + name_hyperparam[i] + " = " + str(most_common(choose_hyperparam[i])))
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()

    
    
    
    
