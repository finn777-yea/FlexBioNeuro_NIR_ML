# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:45:51 2023

@author: ge23hum
"""


from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Classification/, for `common`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `paths`

from paths import calibration_path, data_path

from common.helpers import most_common, warn
from common.data import DataProcessor
from common.features import Scaling_and_FeatureExtractor
from common.nn_hyperparameter_search import PsoHyperparamSearcher
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


#%% Main
def process_fold(train_ix, test_ix, X_features_a, X_features_b, X_features_c, y_all, groups, meas_sec, df_all):
    X_train_val_a, X_test_a = X_features_a[train_ix, :],      X_features_a[test_ix, :]
    X_train_val_b, X_test_b = X_features_b[train_ix, :],      X_features_b[test_ix, :]
    X_train_val_c, X_test_c = X_features_c[train_ix, :],      X_features_c[test_ix, :]
    y_train_val,   y_test   = y_all[train_ix], y_all[test_ix]
        
        
        
        
        
    """
    ######################################################################################################
    # Extract features, Class: Scaling_and_FeatureExtractor
    ######################################################################################################
    """
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
    # Machine Learning, Class: PsoHyperparamSearcher
    ######################################################################################################
    """
    searcher = PsoHyperparamSearcher(
        X_train=features_extractor.X_out,
        y_train=features_extractor.y_train_val,
        X_val=features_extractor.X_out_test,
        y_val=features_extractor.y_test,
        max_num_iteration=10 - 8,
        n_particles=1000 - 995,
        epochs=300,
        model_builder=BaselineNnBuilder(),
    )
    searcher.run()
        
        
        
        
        
    """
    ######################################################################################################
    # Save results
    ######################################################################################################
    """
    name_hyperparam = searcher.best_parameter_names()
    choose_hyperparam = searcher.best_parameter_values()
    mean_scores_train, mean_scores_test = searcher.best_fold_scores()

    print('')
    print('mean_scores_train: %.2f' % mean_scores_train)
    print('mean_scores_test: %.2f' % mean_scores_test)
    print('Best parameters:')
    print(choose_hyperparam)
    print('')

    return name_hyperparam, choose_hyperparam, mean_scores_train, mean_scores_test


if __name__ == "__main__":
    """
    ######################################################################################################
    # Calibration case: VFA/TA-ratio or acetic acid concentration
    ######################################################################################################
    """
    cal_case = "Ac_acid"
    if cal_case == "VFA_TA":
        data_file_path = data_path("classification_NIR_Data_raw_VFA_TA.csv")
        calibrate      = "no"
    elif cal_case == "Ac_acid":
        data_file_path = data_path("classification_NIR_Data_raw_Ac_acid.csv")
        calibrate      = "yes"
    
    
    
    
    
    """
    ######################################################################################################
    # Processing data, Class: DataProcessor
    ######################################################################################################
    """
    data_processor = DataProcessor(data_file_path        = data_file_path,
                                   calibration_file_path = calibration_path(),
                                   meas_sec              = 8)

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
    gkf = StratifiedGroupKFold(n_splits=10)
    
    choose_hyperparam = []
    mean_scores_train = list()
    mean_scores_test  = list()
    name_hyperparam = None

    for train_ix, test_ix in gkf.split(data_processor.X_cal, data_processor.y_all, data_processor.groups):
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
        )
        if not len(choose_hyperparam):
            name_hyperparam = result1
            choose_hyperparam = result2
        else:
            choose_hyperparam = np.column_stack((choose_hyperparam, result2))

        mean_scores_train.append(result3)
        mean_scores_test.append(result4)

        
    print("")
    print("")
    print("Calibration for :  " + cal_case)
    print("Chosen Algorithm:  Deep Neural Network")
    print('mean_scores_train, Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_train)*100, np.std(mean_scores_train)*100))
    print('mean_scores_test,  Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_test)*100, np.std(mean_scores_test)*100))
    
    print("")
    for i in range(len(name_hyperparam)):
        print("Hyperparameter " + name_hyperparam[i] + " = " + str(most_common(choose_hyperparam[i])))

    
    
    
    
