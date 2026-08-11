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
    def __init__(self, X_out, y_train_val, X_out_test, y_test, 
                 max_num_iteration=10, population_size=200, function_timeout=300, epochs=200):
        """
        Initialize the NeuralNetworkOptimizer class with parameters.

        Parameters:
            X_out (numpy.ndarray):       Features of the training and validation data.
            y_train_val (numpy.ndarray): Target values for the training and validation data.
            X_out_test (numpy.ndarray):  Features of the test data.
            y_test (numpy.ndarray):      Target values for the test data.
            max_num_iteration (int):     Maximum number of iterations for the optimization algorithm.
            population_size (int):       Size of the population in the optimization algorithm.
            function_timeout (int):      Maximum time allowed for evaluating the objective function.
            epochs (int):                Number of training epochs for neural network models.

        Returns:
            None: Initializes the class attributes.
        """
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
        self.X_out             = X_out
        self.y_train_val       = y_train_val
        self.X_out_test        = X_out_test
        self.y_test            = y_test
        self.ga_scores_train   = list()
        self.ga_scores_test    = list()
        self.max_num_iteration = max_num_iteration
        self.population_size   = population_size
        self.function_timeout  = function_timeout
        self.epochs            = epochs
        self.evo_algo          = None
        
        
    def create_baseline(self,
                        neuron_layer_1 = 15, 
                        neuron_layer_2 = 8,
                        activation_1   = "elu", 
                        activation_2   = "elu", 
                        activation_3   = "sigmoid",
                        kernel_init    = 7,
                        bias_init      = 5,
                        kernel_reg     = None,
                        bias_reg       = None,
                        activity_reg   = None,
                        kernel_const   = None,
                        bias_const     = None,
                        optim          = "Adam"):
        """
        Create a neural network model with specified parameters.

        Parameters:
            neuron_layer_1 (int): Number of neurons in the first hidden layer.
            neuron_layer_2 (int): Number of neurons in the second hidden layer.
            activation_1 (str):   Activation function for the first hidden layer.
            activation_2 (str):   Activation function for the second hidden layer.
            activation_3 (str):   Activation function for the output layer.
            kernel_init (int):    Index of the kernel initializer in the keras_init list.
            bias_init (int):      Index of the bias initializer in the keras_init list.
            kernel_reg (str):     Regularization for kernel weights.
            bias_reg (str):       Regularization for bias weights.
            activity_reg (str):   Regularization for activity.
            kernel_const (str):   Constraint for kernel weights.
            bias_const (str):     Constraint for bias weights.
            optim (str):          Optimizer for model training.

        Returns:
            keras.models.Sequential: Compiled Keras model.
        """
        # generate model
        model = Sequential()
        model.add(Dense(units                = neuron_layer_1, 
                        input_dim            = np.shape(self.X_out)[1], 
                        activation           = activation_1, 
                        kernel_initializer   = self.keras_init[kernel_init], 
                        bias_initializer     = self.keras_init[bias_init],
                        kernel_regularizer   = kernel_reg,
                        bias_regularizer     = bias_reg,
                        activity_regularizer = activity_reg,
                        kernel_constraint    = kernel_const,
                        bias_constraint      = bias_const))
        
        model.add(Dense(units                = neuron_layer_2, 
                        activation           = activation_2, 
                        kernel_initializer   = self.keras_init[kernel_init], 
                        bias_initializer     = self.keras_init[bias_init],
                        kernel_regularizer   = kernel_reg,
                        bias_regularizer     = bias_reg,
                        activity_regularizer = activity_reg,
                        kernel_constraint    = kernel_const,
                        bias_constraint      = bias_const))
    
        model.add(Dense(units                = 1,
                        activation           = activation_3, 
                        kernel_initializer   = self.keras_init[kernel_init], 
                        bias_initializer     = self.keras_init[bias_init],
                        kernel_regularizer   = kernel_reg,
                        bias_regularizer     = bias_reg,
                        activity_regularizer = activity_reg,
                        kernel_constraint    = kernel_const,
                        bias_constraint      = bias_const))
        
        # compile model
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['binary_accuracy'])
        return model
    
    def objective_function(self, args):
        """
        Calculate the objective function for the given neural network parameters.

        Parameters:
            args (dict): Dictionary of neural network configuration parameters.

        Returns:
            float: The mean squared error of the model's predictions on the validation data.
        """
        estimator = self.create_baseline(**args)
        estimator.fit(x               = self.X_out, 
                      y               = self.y_train_val, 
                      epochs          = self.epochs, 
                      batch_size      = 32,
                      verbose         = 0,
                      validation_data = (self.X_out_test, self.y_test))
        
        
        preds_train  = (estimator.predict(self.X_out) > 0.5).astype("int32")
        preds_test   = (estimator.predict(self.X_out_test) > 0.5).astype("int32")
        
        scores_train = balanced_accuracy_score(self.y_train_val,preds_train)
        scores_test  = balanced_accuracy_score(self.y_test,preds_test)
        
        print("DNN")
        print(f'Score on training set:   balanced_accuracy={scores_train*100:.1f}%')
        print(f'Score on validation set: balanced_accuracy={scores_test*100:.1f}%')
        print(' ')
        print(' ')
        
        # save all scores from GA population
        self.ga_scores_train.append(scores_train)
        self.ga_scores_test.append(scores_test)

        return scores_test * -1 # Expects a value to be minimized
    
    def run_optimization(self):
        """
        Run an optimization algorithm to find the best neural network configuration.
        """
        # Objective parameters
        objective_parameters = [
            {'name'   : 'neuron_layer_1',
             'bounds' : [10, 15],
             'type'   : 'int'},
            
            {'name'   : 'neuron_layer_2',
             'bounds' : [5, 9],
             'type'   : 'int'},
            
            {'name'   : 'activation_1',   
             'bounds' : ["relu", "softmax", "sigmoid", "softplus", "softsign", 
                         "tanh", "selu", "elu", "exponential"],
             'type'   : 'cat'},
            
            {'name'   : 'activation_2',
             'bounds' : ["relu", "softmax", "sigmoid", "softplus", "softsign", 
                         "tanh", "selu", "elu", "exponential"], 
             'type'   : 'cat'},
            
            {'name'   : 'activation_3',
             'bounds' : ["sigmoid"],
             'type'   : 'cat'},
            
            {'name'   : 'kernel_init',
             'bounds' : [0, 9],
             'type'   : 'int'},
            
            {'name'   : 'bias_init',
             'bounds' : [0, 9],
             'type'   : 'int'},
            
            {'name'   : 'kernel_reg',
             'bounds' : ["L1", "L2", "L1L2"],
             'type'   : 'cat'},
            
            {'name'   : 'bias_reg',
             'bounds' : ["L1", "L2", "L1L2"],
             'type'   : 'cat'},
            
            {'name'   : 'activity_reg',
             'bounds' : ["L1", "L2", "L1L2"],
             'type'   : 'cat'},
            
            {'name'   : 'kernel_const',
             'bounds' : ["MaxNorm", "MinMaxNorm", "NonNeg", "UnitNorm"],
             'type'   : 'cat'},
            
            {'name'   : 'bias_const',
             'bounds' : ["MaxNorm", "MinMaxNorm", "NonNeg", "UnitNorm"],
             'type'   : 'cat'},
            
            {'name'   : 'optim',
             'bounds' : ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl",
                         "Nadam", "RMSprop", "SGD"],
             'type'   : 'cat'},
        ]
        
        # Create instance of EA object
        algorithm_parameters = {'max_num_iteration'           : self.max_num_iteration,
                                'population_size'             : self.population_size,
                                'mutation_probability'        : 0.1,
                                'elite_ratio'                 : 0.05,
                                'crossover_probability'       : 0.5,
                                'parents_portion'             : 0.3,
                                'crossover_type'              : 'uniform',
                                'max_iteration_without_improv': None}
        
        self.evo_algo = ea(function             = self.objective_function, 
                           parameters           = objective_parameters,
                           function_timeout     = self.function_timeout,
                           algorithm_parameters = algorithm_parameters)
        
        # Run EA
        self.evo_algo.run()




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
    # Machine Learning, Class: NeuralNetworkOptimizer
    ######################################################################################################
    """
    nn_optimizer = NeuralNetworkOptimizer(X_out             = features_extractor.X_out, 
                                          y_train_val       = features_extractor.y_train_val, 
                                          X_out_test        = features_extractor.X_out_test, 
                                          y_test            = features_extractor.y_test, 
                                          max_num_iteration = 10-8,  # Genetic Algorithm
                                          population_size   = 1000-995, # Genetic Algorithm
                                          function_timeout  = 300, # Genetic Algorithm
                                          epochs            = 300) # Neural Network
    nn_optimizer.run_optimization()
        
        
        
        
        
    """
    ######################################################################################################
    # Save results
    ######################################################################################################
    """
    # Store the results, hyperparameters, and mean scores for this fold
    name_hyperparam   = list(nn_optimizer.evo_algo.best_parameters.keys())
    choose_hyperparam = list(nn_optimizer.evo_algo.best_parameters.values())
    
        
    # save the best results from all GA population
    mean_scores_train = nn_optimizer.ga_scores_train[nn_optimizer.ga_scores_test.index(max(nn_optimizer.ga_scores_test))]
    mean_scores_test  = max(nn_optimizer.ga_scores_test)
    
    return name_hyperparam, choose_hyperparam, mean_scores_train, mean_scores_test


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
    mean_scores_test  = list()
    
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

    
    
    
    