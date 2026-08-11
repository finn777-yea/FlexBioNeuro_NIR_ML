# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:45:51 2023

@author: ge23hum
"""


from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.helpers import most_common, warn
from common.data import DataProcessor
from common.features import Scaling_and_FeatureExtractor
from common.nn_hyperparameter_search import GaHyperparamSearcher
from common.nn_model import BaselineNnBuilder

import concurrent.futures
import random
import warnings

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.utils import set_random_seed

warnings.warn = warn
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
set_random_seed(42)


#%% Main
def process_fold(train_ix, test_ix, X_features_a, X_features_b, X_features_c, y_all, groups, meas_sec, df_all):
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
        
        
        
        
        
    """
    ######################################################################################################
    # Machine Learning, Class: GaHyperparamSearcher
    ######################################################################################################
    """
    searcher = GaHyperparamSearcher(
        X_train=features_extractor.X_out,
        y_train=features_extractor.y_train_val,
        X_val=features_extractor.X_out_test,
        y_val=features_extractor.y_test,
        function_timeout=300,
        epochs=300,
        model_builder=BaselineNnBuilder(),
    )
    searcher.run(max_num_iteration=10 - 8, population_size=1000 - 995)
        
        
        
        
        
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
    cal_case = "VFA_TA"
    if cal_case == "VFA_TA":
        data_file_path = "Data/classification_NIR_Data_raw_VFA_TA.csv"
        calibrate      = "no"#"yes"
    elif cal_case == "Ac_acid":
        data_file_path = "Data/classification_NIR_Data_raw_Ac_acid.csv"
        calibrate      = "yes"
    
    
    
    
    
    """
    ######################################################################################################
    # Processing data, Class: DataProcessor
    ######################################################################################################
    """
    # Initialize Data Processor
    data_processor = DataProcessor(data_file_path        = data_file_path,
                                   calibration_file_path = "Data/calibration_NIR_Data.csv",
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
    gkf = StratifiedGroupKFold(n_splits=5)
    
    # Initialize lists to store hyperparameters and mean scores for training and validation sets
    choose_hyperparam = []
    mean_scores_train = list()
    mean_scores_val  = list()
    
    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # Iterate through cross-validation folds
        futures = []
        for train_ix, test_ix in gkf.split(data_processor.X_cal, data_processor.y_all, data_processor.groups):
            future = executor.submit(process_fold, train_ix, test_ix, X_features_a, X_features_b, X_features_c,
                                     data_processor.y_all, data_processor.groups, data_processor.meas_sec, data_processor.df_all)
            futures.append(future)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

        # Retrieve results from completed futures
        for future in futures:
            result1, result2, result3, result4 = future.result()
            if not len(choose_hyperparam):
                name_hyperparam   = result1
                choose_hyperparam = result2
            else:
                choose_hyperparam = np.column_stack((choose_hyperparam, result2))
            
            mean_scores_train.append(result3)
            mean_scores_val.append(result4)
        del result1, result2, result3, result4
    
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

    
    
    
    