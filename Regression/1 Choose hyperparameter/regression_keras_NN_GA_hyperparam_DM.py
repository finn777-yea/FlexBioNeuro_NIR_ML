# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:45:51 2023

@author: ge23hum
"""

# Repository paths
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `paths`

from paths import calibration_path, data_path

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
from sklearn.tree import DecisionTreeRegressor  # For Decision Tree regression
from sklearn.feature_selection import SelectKBest, f_regression  # For feature selection with chi-squared test
from sklearn.model_selection import StratifiedGroupKFold  # For cross-validation with group information
from sklearn.cross_decomposition import PLSRegression  # For Partial Least Squares regression
from scipy import signal  # For signal processing and detrending

# Machine learning regressors (Deep neural network)
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from evolutionary_algorithm import EvolutionaryAlgorithm as ea
from sklearn.metrics import mean_squared_error, r2_score

import concurrent.futures

# Choose hyperparameter from most common elements
from statistics import mode
def most_common(List):
	return(mode(List))

# To ignore all warnings
def warn(*args, **kwargs):
    pass
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


class DataProcessor:
    def __init__(self, data_file_path, data_grouping_path, calibration_file_path, meas_sec):
        """
        Initialize a DataProcessor object.

        Args:
            data_file_path (str): Path to the data file.
            calibration_file_path (str): Path to the calibration file.
            meas_sec (int): Measurement duration in seconds.
        """
        self.data_file_path        = data_file_path
        self.data_grouping_path    = data_grouping_path
        self.calibration_file_path = calibration_file_path
        self.meas_sec              = meas_sec
        self.df_all                = None
        self.X_all                 = None
        self.y_all                 = None
        self.groups                = None
        self.X_cal                 = None
        self.X_features            = None
        self.df_class              = None
        self.X_class               = None
        self.y_class               = None

    def load_data(self):
        """
        Load preprocessed variables from a CSV data file.
        """
        self.df_all = pd.read_csv(self.data_file_path, encoding='ISO-8859-1')

    def set_XY_values(self):
        """
        Set X and Y values based on the loaded data.
        """
        self.X_all  = self.df_all.iloc[:, :-1].values
        self.y_all  = self.df_all.iloc[:, -1].values
        
    def set_groups(self):
        """
        Set groups based on the loaded data.
        """
        self.df_class = pd.read_csv(self.data_grouping_path, encoding='ISO-8859-1')
        self.X_class  = self.df_class.iloc[:,:-1].values
        self.y_class  = self.df_class.iloc[:,-1].values
        self.groups = list(np.repeat(np.arange(1, int(np.shape(self.X_class)[0] / self.meas_sec) + 1), self.meas_sec))

    def load_calibration(self):
        """
        Load calibration data and calculate absorbance (X_cal).
        """
        df_cal     = pd.read_csv(self.calibration_file_path, encoding='ISO-8859-1')
        cal_dark   = np.array(df_cal.iloc[0, :-1].values, dtype=np.float64)
        cal_ref    = np.array(df_cal.iloc[1, :-1].values, dtype=np.float64)
        self.X_cal = -np.log10((self.X_all - cal_dark) / (cal_ref - cal_dark))
        
    def randomize(self):
        """
        Randomly shuffle the data.
        """
        # Generate a random permutation of indices for data shuffling
        permutation = np.random.permutation(int(self.y_all.shape[0] / self.meas_sec))
        
        # Create a permutation group to apply to both X and Y
        permutation_group = np.repeat(permutation * 0, self.meas_sec)
        
        # Populate the permutation group with corresponding indices
        for i in range(len(permutation)):
            permutation_group[i * self.meas_sec:(i + 1) * self.meas_sec] = np.arange(permutation[i] * self.meas_sec, (permutation[i] + 1) * self.meas_sec)
        
        # Shuffle the data based on the permutation
        self.X_features = self.X_features[permutation_group, :]
        self.y_all = self.y_all[permutation_group]
        
    def msc(self, input_data, reference=None):
        ''' Perform Multiplicative scatter correction
        
        Args:
            input_data (numpy.ndarray): Input data to be corrected.
            reference (numpy.ndarray, optional): Reference spectrum. If not provided, it's estimated from the mean.
        
        Returns:
            tuple: A tuple containing the corrected data and the reference spectrum.
        '''
     
        # Mean center correction
        for i in range(input_data.shape[0]):
            input_data[i,:] -= input_data[i,:].mean()
     
        # Get the reference spectrum. If not given, estimate it from the mean    
        if reference is None:    
            ref = np.mean(input_data, axis=0)
        else:
            ref = reference
     
        # Define a new array and populate it with the corrected data    
        data_msc = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            # Run regression
            fit = np.polyfit(ref, input_data[i,:], 1, full=True)
            # Apply correction
            data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
     
        return (data_msc, ref)

    def snv(self, input_data):
        ''' Perform Standard Normal Variate (SNV) correction
        
        Args:
            input_data (numpy.ndarray): Input data to be corrected.
        
        Returns:
            numpy.ndarray: Corrected data using SNV.
        '''
        # Define a new array and populate it with the corrected data  
        output_data = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            # Apply correction
            output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
     
        return output_data
    
    def split_blocks(self, X):
        ''' Split input data into blocks for different sensors
        
        Args:
            X (numpy.ndarray): Input data to be split.
        
        Returns:
            tuple: A tuple containing separate data blocks for each sensor.
        '''
        # Length of each sensor's data
        len_S1_4 = len(range(1100, 1352, 2))
        len_S1_7 = len(range(1350, 1652, 2))
        len_S2_0 = len(range(1550, 1952, 2))
        len_S2_2 = len(range(1750, 2152, 2))
        
        end_S1_4 = len_S1_4
        end_S1_7 = len_S1_4 + len_S1_7
        end_S2_0 = len_S1_4 + len_S1_7 + len_S2_0
        end_S2_2 = len_S1_4 + len_S1_7 + len_S2_0 + len_S2_2
        
        # Split data for each sensor
        X_S1_4   = X[:, 0:end_S1_4]
        X_S1_7   = X[:, end_S1_4:end_S1_7]
        X_S2_0   = X[:, end_S1_7:end_S2_0]
        X_S2_2   = X[:, end_S2_0:end_S2_2]
        
        return X_S1_4, X_S1_7, X_S2_0, X_S2_2
    
    def baseline_correction(self, method):
        """
        Apply baseline correction to data.

        Args:
            method (int): The method to use for baseline correction.
        """
        X_S1_4, X_S1_7, X_S2_0, X_S2_2 = self.split_blocks(copy.deepcopy(self.X_cal))
        
        if method == 1:
            # Detrending data
            X_S1_4 = signal.detrend(X_S1_4, axis=1)
            X_S1_7 = signal.detrend(X_S1_7, axis=1)
            X_S2_0 = signal.detrend(X_S2_0, axis=1)
            X_S2_2 = signal.detrend(X_S2_2, axis=1)
        elif method == 2:
            # Subtract mean from the data
            X_S1_4 = X_S1_4 - np.tile(np.mean(X_S1_4, axis=1), (X_S1_4.shape[1], 1)).T
            X_S1_7 = X_S1_7 - np.tile(np.mean(X_S1_7, axis=1), (X_S1_7.shape[1], 1)).T
            X_S2_0 = X_S2_0 - np.tile(np.mean(X_S2_0, axis=1), (X_S2_0.shape[1], 1)).T
            X_S2_2 = X_S2_2 - np.tile(np.mean(X_S2_2, axis=1), (X_S2_2.shape[1], 1)).T
        elif method == 3:
            # SNV
            for i in range(int(self.X_cal.shape[0] / self.meas_sec)):
                X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :])
        elif method == 4:
            # MSC
            for i in range(int(self.X_cal.shape[0] / self.meas_sec)):
                (X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :])
        
        # Combine the data again
        self.X_features = np.hstack((X_S1_4, X_S1_7, X_S2_0, X_S2_2))
        



class Scaling_and_FeatureExtractor:
    def __init__(self, number_of_features, X_train_val, y_train_val, X_test, y_test, df_all):
        """
        Initialize a Scaling_and_FeatureExtractor object.

        Args:
            number_of_features (int):    Number of features to select.
            X_train_val (numpy.ndarray): Training and validation input data.
            y_train_val (numpy.ndarray): Training and validation output data.
            X_test (numpy.ndarray):      Test input data.
            y_test (numpy.ndarray):      Test output data.
            df_all (pd.DataFrame):       Dataframe containing all features.
        """
        self.number_of_features = number_of_features
        self.X_train_val        = X_train_val
        self.y_train_val        = y_train_val
        self.X_test             = X_test
        self.y_test             = y_test
        self.df_all             = df_all
        self.X_out              = None
        self.X_out_test         = None
        self.X_ml               = None
        self.y_ml               = None
        self.ps                 = None
        
    def Scaling_data(self):
        """
        Scale input data using MinMaxScaler.
        """
        scaler_train_val = MinMaxScaler(feature_range=(0, 1))
        self.X_train_val = scaler_train_val.fit_transform(self.X_train_val)
        self.X_test      = scaler_train_val.transform(self.X_test)
        

    def extra_trees_feature_reduction(self, X_train_val, y_train_val, df_all, X_test):
        """
        Perform feature reduction using Extra Trees Classifier.

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            y_train_val (numpy.ndarray): Training and validation output data.
            df_all (pd.DataFrame):       Dataframe containing all features.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data and the selected feature names.
        """
        # Feature reduction using Extra Trees Classifier
        model                   = DecisionTreeRegressor()
        model.fit(X_train_val, y_train_val)
        feat_importances        = pd.Series(model.feature_importances_, index=df_all.columns[:-1])

        # take few best wavelengths (dfcolumns_out) and 
        # save these measurements (X_out) from these best wavelengths
        dfcolumns_out           = []
        best_feature_importance = feat_importances.nlargest(self.number_of_features)
        ind_best_feature        = np.argwhere(df_all.columns[:-1]==best_feature_importance.index[0])[0][0]
        X_out_FR                = copy.deepcopy(X_train_val[:,ind_best_feature])
        X_out_test_FR           = copy.deepcopy(X_test[:,ind_best_feature])
        dfcolumns_out.append(df_all.columns[ind_best_feature])
        for i in range(self.number_of_features-1):
            ind_best_feature = np.argwhere(df_all.columns[:-1]==best_feature_importance.index[i+1])[0][0]
            X_out_FR         = np.column_stack((X_out_FR, X_train_val[:,ind_best_feature]))
            X_out_test_FR    = np.column_stack((X_out_test_FR, X_test[:,ind_best_feature]))
            dfcolumns_out.append(df_all.columns[ind_best_feature])

        return X_out_FR, X_out_test_FR, dfcolumns_out

    def univariate_selection(self, X_train_val, y_train_val, df_all, X_test):
        """
        Perform feature selection using the f_regression test.

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            y_train_val (numpy.ndarray): Training and validation output data.
            df_all (pd.DataFrame):       Dataframe containing all features.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data and the selected feature names.
        """
        # Univariate feature selection using chi-squared test
        bestfeatures              = SelectKBest(score_func=f_regression, k=self.number_of_features)
        fit                       = bestfeatures.fit(X_train_val, y_train_val)
        dfcolumns                 = pd.DataFrame(df_all.columns[:-1])
        dfscores                  = pd.DataFrame(fit.scores_)

        # Concatenate two dataframes for better visualization
        featureScores             = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns     = ['Specs', 'Score']

        # take 10 best wavelengths (dfcolumns_out) and 
        # save these measurements (X_out) from these best wavelengths
        dfcolumns_out           = []
        best_univariate_selection = featureScores.nlargest(self.number_of_features,'Score')
        # -----------------------------------------------------------
        ind_best_feature        = best_univariate_selection.index[0]
        X_out_US                = copy.deepcopy(X_train_val[:,ind_best_feature])
        X_out_test_US           = copy.deepcopy(X_test[:,ind_best_feature])
        dfcolumns_out.append(df_all.columns[ind_best_feature])
        for i in range(1,self.number_of_features):
            ind_best_feature = best_univariate_selection.index[i]
            X_out_US         = np.column_stack((X_out_US, X_train_val[:,ind_best_feature]))
            X_out_test_US    = np.column_stack((X_out_test_US, X_test[:,ind_best_feature]))
            dfcolumns_out.append(df_all.columns[ind_best_feature])

        return X_out_US, X_out_test_US, dfcolumns_out

    def pls(self, X_train_val, y_train_val, X_test):
        """
        Perform feature reduction using Partial Least Squares (PLS).

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            y_train_val (numpy.ndarray): Training and validation output data.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data.
        """
        # Partial Least Squares (PLS) feature reduction
        plsda          = PLSRegression(n_components=self.number_of_features)
        (X_out_PLS,_)  = plsda.fit_transform(X_train_val, y_train_val)
        X_out_test_PLS = plsda.transform(X_test)

        return X_out_PLS, X_out_test_PLS

    def pca(self, X_train_val, X_test):
        """
        Perform feature reduction using Principal Component Analysis (PCA).

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data.
        """
        # Principal Component Analysis (PCA) feature reduction
        sklearn_pca    = PCA(n_components=self.number_of_features)
        X_out_PCA      = sklearn_pca.fit_transform(X_train_val)
        X_out_test_PCA = sklearn_pca.transform(X_test)

        return X_out_PCA, X_out_test_PCA

    def truncated_svd(self, X_train_val, X_test):
        """
        Perform feature reduction using Truncated Singular Value Decomposition (TruncatedSVD).

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data.
        """
        # Truncated Singular Value Decomposition (TruncatedSVD) feature reduction
        sklearn_svd    = TruncatedSVD(n_components=self.number_of_features)
        X_out_SVD      = sklearn_svd.fit_transform(X_train_val)
        X_out_test_SVD = sklearn_svd.transform(X_test)

        return X_out_SVD, X_out_test_SVD

    def extract_features(self):
        """
        Extract and reduce features using various methods.
        """
        X_out_FR,  X_out_test_FR, _ = self.extra_trees_feature_reduction(self.X_train_val, self.y_train_val, self.df_all, self.X_test)
        X_out_US,  X_out_test_US, _ = self.univariate_selection(self.X_train_val, self.y_train_val, self.df_all, self.X_test)
        X_out_PLS, X_out_test_PLS   = self.pls(self.X_train_val, self.y_train_val, self.X_test)
        X_out_PCA, X_out_test_PCA   = self.pca(self.X_train_val, self.X_test)
        X_out_SVD, X_out_test_SVD   = self.truncated_svd(self.X_train_val, self.X_test)

        # Combine the selected features from different methods
        self.X_out      = np.hstack((X_out_FR,      X_out_US,      X_out_PLS,      X_out_PCA,      X_out_SVD))
        self.X_out_test = np.hstack((X_out_test_FR, X_out_test_US, X_out_test_PLS, X_out_test_PCA, X_out_test_SVD))
        
    def Predefined_cv(self):
        """
        Perform predefined cross-validation.
        """
        # The indices which have the value -1 will be kept in train.
        train_indices = np.full((np.size(self.y_train_val),), -1, dtype=int)
    
        # The indices which have zero or positive values, will be kept in test
        val_indices   = np.full((np.size(self.y_test),), 0, dtype=int)
        test_fold     = np.append(train_indices, val_indices)
    
        # Predefined
        self.ps       = PredefinedSplit(test_fold)
        self.X_ml     = np.concatenate((self.X_out,       self.X_out_test))
        self.y_ml     = np.concatenate((self.y_train_val, self.y_test))
    



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
                        activation_3   = "linear",
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
        model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mean_absolute_error'])
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
        
        
        preds_train  = estimator.predict(self.X_out)
        preds_test   = estimator.predict(self.X_out_test)
        
        if np.isnan(np.array(preds_train)).any() or np.isnan(np.array(preds_test)).any():
            scores_train   = 1e12 # very big, infinity
            scores_test    = 1e12 # very big, infinity
            
            r2_score_train = 0
            r2_score_test  = 0
            
        else:
            scores_train = mean_squared_error(self.y_train_val,preds_train,squared=False)
            scores_test  = mean_squared_error(self.y_test,preds_test,squared=False)
            
            r2_score_train = r2_score(self.y_train_val,preds_train)
            r2_score_test  = r2_score(self.y_test,preds_test)
        
        print("DNN")
        print(f'Score on training set:   mean_squared_error = {scores_train:.1f}')
        print(f'Score on validation set: mean_squared_error = {scores_test:.1f}')
        print(f'Score on training set:   r2_score           = {r2_score_train*100:.1f}%')
        print(f'Score on validation set: r2_score           = {r2_score_test*100:.1f}%')
        print(' ')
        print(' ')
        
        # save all scores from GA population
        self.ga_scores_train.append(scores_train)
        self.ga_scores_test.append(scores_test)

        return scores_test # Expects a value to be minimized
    
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
             'bounds' : ["linear", "relu"],
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
def process_fold(train_ix, test_ix, X_features, y_all, groups, meas_sec, df_all):
    # Split data for the current fold
    X_train_val, X_test = X_features[train_ix, :], X_features[test_ix, :]
    y_train_val, y_test = y_all[train_ix],         y_all[test_ix]
    
    
    
    
    
    """
    ######################################################################################################
    # Features Extraction, Class: Scaling_and_FeatureExtractor
    ######################################################################################################
    """
    # Feature Extraction and Predefined Cross-Validation Setup
    features_extractor = Scaling_and_FeatureExtractor(number_of_features = 5, 
                                                      X_train_val        = X_train_val, 
                                                      y_train_val        = y_train_val, 
                                                      X_test             = X_test, 
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
                                          function_timeout  = 400, # Genetic Algorithm
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
    mean_scores_train = nn_optimizer.ga_scores_train[nn_optimizer.ga_scores_test.index(min(nn_optimizer.ga_scores_test))]
    mean_scores_test  = min(nn_optimizer.ga_scores_test)
        
    return name_hyperparam, choose_hyperparam, mean_scores_train, mean_scores_test



if __name__ == "__main__":
    """
    ######################################################################################################
    # Processing data, Class: DataProcessor
    ######################################################################################################
    """
    # Initialize Data Processor
    data_processor = DataProcessor(data_file_path        = data_path("regression_NIR_Data_raw_DM.csv"),
                                   data_grouping_path    = data_path("classification_NIR_Data_raw_DM.csv"),
                                   calibration_file_path = calibration_path(),
                                   meas_sec              = 8)

    # Load raw data, set up features and targets, load calibration data, and perform baseline correction
    data_processor.load_data()
    data_processor.set_XY_values()
    data_processor.set_groups()
    data_processor.load_calibration()
    data_processor.baseline_correction(method=2)
    data_processor.randomize()
    
    
    
    
    
    """
    ######################################################################################################
    # FOR-loop preparation
    ######################################################################################################
    """
    # Initialize Stratified Group K-Fold for cross-validation
    gkf = StratifiedGroupKFold(n_splits=10)
    
    # Initialize lists to store hyperparameters and mean scores for training and validation sets
    choose_hyperparam     = []
    mean_scores_train     = list()
    mean_scores_test      = list()

    
    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        # Iterate through cross-validation folds
        futures = []
        for train_ix, test_ix in gkf.split(data_processor.X_class, data_processor.y_class, data_processor.groups):
            future = executor.submit(process_fold, train_ix, test_ix, data_processor.X_features,
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
    print("")
    print("Algorithm: Neural Network")
    print('mean_scores_train, mean_squared_error: %.2f (%.2f)' % (np.mean(mean_scores_train), np.std(mean_scores_train)))
    print('mean_scores_test,  mean_squared_error: %.2f (%.2f)' % (np.mean(mean_scores_test), np.std(mean_scores_test)))
    
    
    
    print("")
    for i in range(len(name_hyperparam)):
        print("Hyperparameter " + name_hyperparam[i] + " = " + str(most_common(choose_hyperparam[i])))

    
    
    
    