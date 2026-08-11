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

# Data manipulation and analysis
import pandas as pd  # For data handling and analysis
import numpy as np  # For numerical operations
import copy  # For creating deep copies of data

# Data visualization
import matplotlib.pyplot as plt  # For creating plots and visualizations
import matplotlib.patches as mpatches  # For creating patches in plots
import matplotlib as mpl  # For customizing Matplotlib behavior

# Data preprocessing and machine learning
from sklearn.preprocessing import MinMaxScaler  # For feature scaling
from sklearn.model_selection import PredefinedSplit  # For cross-validation and hyperparameter tuning
from sklearn.decomposition import PCA, TruncatedSVD  # For dimensionality reduction
from sklearn.ensemble import ExtraTreesClassifier  # For Extra Trees classification
from sklearn.feature_selection import SelectKBest, f_classif  # For feature selection with chi-squared test
from sklearn.model_selection import StratifiedGroupKFold  # For cross-validation with group information
from sklearn.cross_decomposition import PLSRegression  # For Partial Least Squares regression
from scipy import signal  # For signal processing and detrending

# Machine learning classifiers (Deep neural network)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import initializers
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.metrics import balanced_accuracy_score

import concurrent.futures

# Choose hyperparameter from most common elements

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

# to create the same random sequence every time
np.random.seed(42)

import random
random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.utils import set_random_seed
set_random_seed(42)

class NeuralNetworkOptimizer:
    def __init__(self, X_out, y_train_val, X_out_test, y_test, meas_sec,
                 max_num_iteration=1000, n_particles=200, dimensions=12, epochs=200):
        # List of initializers
        self.keras_init       = [initializers.RandomNormal(seed=42),
                                 initializers.RandomUniform(seed=42),
                                 initializers.TruncatedNormal(seed=42),
                                 initializers.VarianceScaling(seed=42),
                                 initializers.GlorotNormal(seed=42),
                                 initializers.GlorotUniform(seed=42),
                                 initializers.HeNormal(seed=42),
                                 initializers.HeUniform(seed=42),
                                 initializers.LecunNormal(seed=42),
                                 initializers.LecunUniform(seed=42)]
        self.activation        = ["relu", "softmax", "sigmoid", "softplus", "softsign", 
                                  "tanh", "selu", "elu", "exponential"]
        self.regulation        = ["L1", "L2", "L1L2", None]
        self.constraint        = ["MaxNorm", "MinMaxNorm", "NonNeg", "UnitNorm", None]
        self.optimizer         = ["Adadelta", "Adagrad", "Adam", "Adamax", 
                                  "Ftrl", "Nadam", "RMSprop", "SGD"]
        self.X_out             = X_out
        self.y_train_val       = y_train_val
        self.X_out_test        = X_out_test
        self.y_test            = y_test
        self.ga_scores_train   = list()
        self.ga_scores_test    = list()
        self.max_num_iteration = max_num_iteration
        self.n_particles       = n_particles
        self.dimensions        = dimensions
        self.epochs            = epochs
        self.meas_sec          = meas_sec
        self.evo_algo          = None
        self.cost              = None
        self.pos               = None
        
        self.name_hyperparam   = ["LSTM_layer",
                                  "activation_1",
                                  "activation_2",
                                  "kernel_init",
                                  "bias_init",
                                  "kernel_reg",
                                  "bias_reg",
                                  "activity_reg",
                                  "kernel_const",
                                  "bias_const",
                                  "optim"]


    def create_baseline(self, x):
        LSTM_layer     = np.round(x[0]).astype(int)
        activation_1   = np.round(x[1]).astype(int)
        activation_2   = np.round(x[2]).astype(int)
        kernel_init    = np.round(x[3]).astype(int)
        bias_init      = np.round(x[4]).astype(int)
        kernel_reg     = np.round(x[5]).astype(int)
        bias_reg       = np.round(x[6]).astype(int)
        activity_reg   = np.round(x[7]).astype(int)
        kernel_const   = np.round(x[8]).astype(int)
        bias_const     = np.round(x[9]).astype(int)
        optim          = np.round(x[10]).astype(int)
        
        # generate model
        model = Sequential()
        model.add(LSTM(units                 = LSTM_layer, 
                       input_shape           = (self.meas_sec, self.X_out.shape[2]), 
                       activation            = self.activation[activation_1],
                       recurrent_activation  = self.activation[activation_2],
                       kernel_initializer    = self.keras_init[kernel_init],
                       recurrent_initializer = self.keras_init[kernel_init],
                       bias_initializer      = self.keras_init[bias_init],
                       kernel_regularizer    = self.regulation[kernel_reg],
                       bias_regularizer      = self.regulation[bias_reg],
                       activity_regularizer  = self.regulation[activity_reg],
                       kernel_constraint     = self.constraint[kernel_const],
                       bias_constraint       = self.constraint[bias_const],
                       return_sequences      = False))
    
        model.add(Dense(units                = 1,
                        activation           = "sigmoid", 
                        kernel_initializer   = self.keras_init[kernel_init], 
                        bias_initializer     = self.keras_init[bias_init],
                        kernel_regularizer   = self.regulation[kernel_reg],
                        bias_regularizer     = self.regulation[bias_reg],
                        activity_regularizer = self.regulation[activity_reg],
                        kernel_constraint    = self.constraint[kernel_const],
                        bias_constraint      = self.constraint[bias_const]))
        
        # compile model
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer[optim], metrics=['binary_accuracy'])
        return model
    
    
    def objective_function(self, args):
        f = list()
        for i in range(self.n_particles):
            estimator = self.create_baseline(args[i,:])
            estimator.fit(x               = self.X_out, 
                          y               = self.y_train_val, 
                          epochs          = self.epochs, 
                          batch_size      = 32,
                          verbose         = 0,
                          validation_data = (self.X_out_test, self.y_test))
            
            
            preds_train  = (estimator.predict(self.X_out) > 0.5).astype("int32")
            preds_test   = (estimator.predict(self.X_out_test) > 0.5).astype("int32")
            
            if np.isnan(np.array(preds_train)).any() or np.isnan(np.array(preds_test)).any():
                scores_train = -10 # very big, infinity
                scores_test  = -10 # very big, infinity
            else:
                scores_train = balanced_accuracy_score(self.y_train_val[:,0],preds_train[:,0])
                scores_test  = balanced_accuracy_score(self.y_test[:,0],preds_test[:,0])
            
            print("DNN")
            print(f'Score on training set:   balanced_accuracy={scores_train*100:.1f}%')
            print(f'Score on validation set: balanced_accuracy={scores_test*100:.1f}%')
            print(' ')
            print(' ')
            
            # save all scores from GA population
            self.ga_scores_train.append(scores_train)
            self.ga_scores_test.append(scores_test)
            
            f.append (scores_test * -1) # Expects a value to be minimized

        return f
    
    
    def run_optimization(self):
        from pyswarms.single.global_best import GlobalBestPSO
        # instatiate the optimizer
        x_max     = np.array([10, 8, 8, 9, 9, 3, 3, 3, 4, 4, 7])
        x_min     = np.array([3,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        bounds    = (x_min, x_max)
        options   = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = GlobalBestPSO(n_particles=self.n_particles, dimensions=self.dimensions, 
                                  options=options, bounds=bounds)
        
        # now run the optimization
        self.cost, self.pos = optimizer.optimize(self.objective_function, self.max_num_iteration, verbose=False)
        
    def best_parameters(self):
        best_param_values = [np.round(self.pos[0]).astype(str),
                             self.activation[np.round(self.pos[1]).astype(int)],
                             self.activation[np.round(self.pos[2]).astype(int)],
                             self.keras_init[np.round(self.pos[3]).astype(int)],
                             self.keras_init[np.round(self.pos[4]).astype(int)],
                             self.regulation[np.round(self.pos[5]).astype(int)],
                             self.regulation[np.round(self.pos[6]).astype(int)],
                             self.regulation[np.round(self.pos[7]).astype(int)],
                             self.constraint[np.round(self.pos[8]).astype(int)],
                             self.constraint[np.round(self.pos[9]).astype(int)],
                             self.optimizer[np.round(self.pos[10]).astype(int)]]
        return best_param_values





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
                                                      df_all             = df_all,
                                                      meas_sec           = meas_sec)
        
    features_extractor.Scaling_data()
    features_extractor.extract_features()
    features_extractor.reshape_data()
        
        
        
        
        
    """
    ######################################################################################################
    # Machine Learning, Class: NeuralNetworkOptimizer
    ######################################################################################################
    """
    nn_optimizer = NeuralNetworkOptimizer(X_out             = features_extractor.X_out, 
                                          y_train_val       = features_extractor.y_train_val, 
                                          X_out_test        = features_extractor.X_out_test, 
                                          y_test            = features_extractor.y_test,
                                          meas_sec          = meas_sec,
                                          max_num_iteration = 10-8,
                                          n_particles       = 1000-995,
                                          dimensions        = 11,
                                          epochs            = 500)
    
    nn_optimizer.run_optimization()
        
        
        
        
        
    """
    ######################################################################################################
    # Save results
    ######################################################################################################
    """
    # Store the results, hyperparameters, and mean scores for this fold
    choose_hyperparam = nn_optimizer.best_parameters()
    name_hyperparam   = nn_optimizer.name_hyperparam   
        
    # save the best results from all GA population
    mean_scores_train = nn_optimizer.ga_scores_train[nn_optimizer.ga_scores_test.index(max(nn_optimizer.ga_scores_test))]
    mean_scores_test  = max(nn_optimizer.ga_scores_test)
        
    # print best results for the current fold
    print('')
    print('mean_scores_train: %.2f' % mean_scores_train)
    print('mean_scores_test: %.2f' % mean_scores_test)
    print('Best parameters:')
    print(nn_optimizer.best_parameters())
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
        data_file_path = "Data/classification_NIR_Data_raw_VFA_TA.csv"
        calibrate      = "no"
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
    gkf = StratifiedGroupKFold(n_splits=10)
    
    # Initialize lists to store hyperparameters and mean scores for training and validation sets
    choose_hyperparam = []
    mean_scores_train = list()
    mean_scores_test  = list()
    
    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
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
            mean_scores_test.append(result4)
        del result1, result2, result3, result4
        
        
    # Print results
    print("")
    print("")
    print("Calibration for :  " + cal_case)
    print("Chosen Algorithm:  Deep Neural Network")
    print('mean_scores_train, Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_train)*100, np.std(mean_scores_train)*100))
    print('mean_scores_test,  Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_test)*100, np.std(mean_scores_test)*100))
    
    print("")
    for i in range(len(name_hyperparam)):
        print("Hyperparameter " + name_hyperparam[i] + " = " + str(most_common(choose_hyperparam[i])))

    
    
    
    