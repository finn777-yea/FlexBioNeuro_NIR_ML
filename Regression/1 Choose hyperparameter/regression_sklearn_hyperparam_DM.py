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
from sklearn.model_selection import PredefinedSplit, GridSearchCV  # For cross-validation and hyperparameter tuning
from sklearn.decomposition import PCA, TruncatedSVD  # For dimensionality reduction
from sklearn.tree import DecisionTreeRegressor  # For Decision Tree regression
from sklearn.feature_selection import SelectKBest, f_regression  # For feature selection with chi-squared test
from sklearn.model_selection import StratifiedGroupKFold, cross_validate  # For cross-validation with group information
from sklearn.cross_decomposition import PLSRegression  # For Partial Least Squares regression
from scipy import signal  # For signal processing and detrending

# Machine learning regressors
from sklearn.neighbors import KNeighborsRegressor  # For k-Nearest Neighbors regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge # For linear regression
import sklearn.gaussian_process as gp # For Gaussian regressor
from sklearn.svm import SVR  # For Support Vector Regression
from sklearn.ensemble import RandomForestRegressor  # For Random Forest regression
from sklearn.ensemble import ExtraTreesRegressor  # For Extra Trees regression
from sklearn.ensemble import GradientBoostingRegressor  # For Gradient Boosting regression
from sklearn.ensemble import AdaBoostRegressor  # For AdaBoost regression
from sklearn.neural_network import MLPRegressor  # For Multi-layer Perceptron regression


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
    



class MLAlgorithm:
    def __init__(self, X_ml, y_ml, ps, n_jobs):
        """
        Initialize an MLAlgorithm object.

        Args:
            X_ml (numpy.ndarray): Input data for machine learning.
            y_ml (numpy.ndarray): Output data for machine learning.
            ps (PredefinedSplit): Predefined cross-validation split.
            n_jobs (int):         Number of CPU cores to use for parallel computation.
        """
        self.X_ml             = X_ml
        self.y_ml             = y_ml
        self.groups           = None
        self.ps               = ps
        self.n_jobs           = n_jobs
        self.grid_result      = None
        self.scores_train     = None
        self.scores_test      = None
        self.mse_scores_train = None
        self.mse_scores_test  = None

    def run_knn(self):
        """
        Run k-Nearest Neighbors Regressor (k-NN) algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_neighbors" : [2, 3, 4, 5, 6, 7],
            "weights"     : ["uniform", "distance"],
            "algorithm"   : ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size"   : [5, 10, 15],
            "p"           : [1, 2]
            }
        
        self.machine_learning_grid_search(KNeighborsRegressor(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "KNeighborsRegressor")

    def run_lin_reg(self):
        """
        Run Linear Regression algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "fit_intercept" : [True, False],
            "copy_X"        : [True, False],
            "positive"      : [True, False]
            }

        self.machine_learning_grid_search(LinearRegression(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "LinearRegression")

    def run_lasso(self):
        """
        Run Laggo Regression algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "alpha"         : [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            "fit_intercept" : [True, False],
            "max_iter"      : [800, 900, 1000, 1100, 1200, 2000, 5000],
            "tol"           : [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            "selection"     : ["cyclic", "random"]
            }

        self.machine_learning_grid_search(Lasso(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "Lasso")
        
    def run_ridge(self):
        """
        Run Ridge Regression algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "alpha"         : [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            "fit_intercept" : [True, False],
            "max_iter"      : [5000],
            "tol"           : [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            "solver"        : ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
            }

        self.machine_learning_grid_search(Ridge(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "Ridge")
    
    def run_gp(self):
        """
        Run Gaussian Regressor algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "alpha"               : [1e-8, 1e-9, 1e-10, 1e-11, 1e-12],
            "n_restarts_optimizer": [0, 1, 2, 3, 4],
            "normalize_y"         : [True, False],
            }

        self.machine_learning_grid_search(gp.GaussianProcessRegressor(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "Gaussian Regressor")
        
    def run_SVR(self):
        """
        Run Support Vector Regressor algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "kernel"  : ["linear", "poly", "rbf", "sigmoid"],
            "degree"  : [2, 3, 4],
            "gamma"   : ["scale", "auto"],
            "tol"     : [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            "verbose" : [False],
            "C"       : [0.8, 0.9, 1.0, 1.1, 1.2],
            "epsilon" : [0.08, 0.09, 0.10, 0.11, 0.12]
            }

        self.machine_learning_grid_search(SVR(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "SVR")
        
    def run_decision_tree(self):
        """
        Run Decision Tree Regressor algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "criterion"        : ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "splitter"         : ["best", "random"],
            "max_depth"        : [None, 5, 6, 7, 8, 9, 10],
            "min_samples_split": [2, 3, 4, 5, 6, 7, 8],
            "min_samples_leaf" : [1, 2, 3, 4, 5, 6, 7],
            "max_features"     : ["sqrt", "log2", None],
            "random_state"     : [42],
            }

        self.machine_learning_grid_search(DecisionTreeRegressor(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "DecisionTreeRegressor")
        
    def run_random_forest(self):
        """
        Run Random Forest Regressor algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_estimators"      : [3],
            "criterion"         : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
            "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9],
            "min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7, 8],
            "max_features"      : ["sqrt", "log2", None],
            "bootstrap"         : [True, False],
            "warm_start"        : [True, False],
            "random_state"      : [42],
            }
        
        self.machine_learning_grid_search(RandomForestRegressor(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "RandomForestRegressor")
        
    def run_extra_trees(self):
        """
        Run Extra Trees Regressor algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_estimators"      : [3],
            "criterion"         : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
        	"min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
        	"max_features"      : [None, "sqrt", "log2"],
        	"bootstrap"         : [True, False],
        	"warm_start"        : [True, False],
            "random_state"      : [42]
            }
        
        self.machine_learning_grid_search(ExtraTreesRegressor(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "ExtraTreesRegressor")
        
    def run_grad_boosting(self):
        """
        Run Gradient Boosting Regressor algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "loss"              : ["squared_error", "absolute_error", "huber", "quantile"],
            "learning_rate"     : [0.01, 0.1, 1.0],
            "n_estimators"      : [3],
            "criterion"         : ["friedman_mse", "squared_error"],
            "min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
            "max_features"      : [None, "sqrt", "log2"],
            "max_leaf_nodes"    : [None],
            "warm_start"        : [True, False],
        	"tol"               : [1e-3, 1e-4, 1e-5],
        	"n_iter_no_change"  : [None],
            "random_state"      : [42]
            }
        
        self.machine_learning_grid_search(GradientBoostingRegressor(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "GradientBoostingRegressor")
        
    def run_ada_boost(self):
        """
        Run Ada Boost Regressor algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "learning_rate" : [0.01, 0.1, 1.0, 10.0],
            "n_estimators"  : [150],
        	"loss"          : ["linear", "square", "exponential"],
            "random_state"  : [42]
            }
        
        self.machine_learning_grid_search(AdaBoostRegressor(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "AdaBoostRegressor")
        
    def run_MLP(self):
        """
        Run Multiple Layer Perceptor Regressor algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "hidden_layer_sizes" : [(10,3), (11,3), (12,3)],
            "activation"         : ["identity", "logistic", "tanh", "relu"],
            "solver"             : ["lbfgs", "sgd", "adam"],
            "learning_rate"      : ["constant", "invscaling", "adaptive"],
            "random_state"       : [42],
            "max_iter"           : [5000],
            "alpha"              : [0.000001,0.00001,0.0001,0.001,0.01],
            }
        
        self.machine_learning_grid_search(MLPRegressor(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "MLPRegressor")
        

    def machine_learning_grid_search(self, model, parameters):
        """
        Perform grid search for hyperparameter tuning.

        Args:
            model:             A machine learning model.
            parameters (dict): Hyperparameters and their possible values.
        """
        # Search Grid
        grid = GridSearchCV(estimator  = model,
                            param_grid = parameters,
                            cv         = self.ps,
                            n_jobs     = self.n_jobs,
                            scoring    = 'neg_mean_squared_error',
                            refit      = True)
        self.grid_result = grid.fit(self.X_ml, self.y_ml, groups=self.groups)

        # Summarize results
        print("Best: %f using %s" % (self.grid_result.best_score_, self.grid_result.best_params_))

    def machine_learning_best_scores(self, mod_best_param, model_name):
        """
        Calculate and print best scores for the selected machine learning model.

        Args:
            mod_best_param:   The best machine learning model obtained from hyperparameter tuning.
            model_name (str): The name of the machine learning model.
        """
        scores = cross_validate(mod_best_param, self.X_ml, self.y_ml, cv=self.ps, groups=self.groups,
                                scoring=('r2', 'neg_mean_squared_error'),
                                return_train_score=True)

        self.scores_train     = np.mean(scores['train_r2'])
        self.scores_test      = np.mean(scores['test_r2'])
        self.mse_scores_train = np.mean(scores['train_neg_mean_squared_error'])
        self.mse_scores_test  = np.mean(scores['test_neg_mean_squared_error'])

        print(model_name)
        print(f'Score on training set:   r2={self.scores_train * 100:.1f}%')
        print(f'Score on validation set: r2={self.scores_test * 100:.1f}%')
        print(f'Score on training set:   MSE={self.mse_scores_train:.1f}')
        print(f'Score on validation set: MSE={self.mse_scores_test:.1f}')
        print(' ')



#%% Main
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
    mean_MSE_scores_train = list()
    mean_MSE_scores_test  = list()
    
    # Iterate through cross-validation folds
    for train_ix, test_ix in gkf.split(data_processor.X_class,data_processor.y_class,data_processor.groups):
        # Split data for the current fold
        X_train_val, X_test = data_processor.X_features[train_ix, :], data_processor.X_features[test_ix, :]
        y_train_val, y_test = data_processor.y_all[train_ix],         data_processor.y_all[test_ix]
    
    
    
    
    
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
                                                          df_all             = data_processor.df_all)
        
        features_extractor.Scaling_data()
        features_extractor.extract_features()
        features_extractor.Predefined_cv()
        
        
        
        
        
        """
        ######################################################################################################
        # Machine Learning, Class: MLAlgorithm
        ######################################################################################################
        """
        # Machine Learning Algorithm Setup
        ml_algorithm = MLAlgorithm(X_ml   = features_extractor.X_ml, 
                                   y_ml   = features_extractor.y_ml, 
                                   ps     = features_extractor.ps, 
                                   n_jobs = 6)
    
        # Initialize the dictionary to map algorithm names to functions
        algorithm_mapping = {
            "knn":           ml_algorithm.run_knn,
            "linear_reg":    ml_algorithm.run_lin_reg,
            "lasso":         ml_algorithm.run_lasso,
            "ridge":         ml_algorithm.run_ridge,
            "gaussian":      ml_algorithm.run_gp,
            "SVR":           ml_algorithm.run_SVR,
            "decision_tree": ml_algorithm.run_decision_tree,
            "random_forest": ml_algorithm.run_random_forest,
            "extra_trees":   ml_algorithm.run_extra_trees,
            "grad_boosting": ml_algorithm.run_grad_boosting,
            "ada_boost":     ml_algorithm.run_ada_boost,
            "MLP":           ml_algorithm.run_MLP
        }
        
        # Specify the chosen algorithm (e.g., "knn", "decision_tree", "MLP")
        chosen_algorithm = "extra_trees"  # Change this to select the desired algorithm
    
        # Check if the chosen algorithm exists in the mapping, then call it
        if chosen_algorithm in algorithm_mapping:
            algorithm_mapping[chosen_algorithm]()
        else:
            print(f"Algorithm '{chosen_algorithm}' is not recognized.")
        
        
        
        
        
        """
        ######################################################################################################
        # Save results
        ######################################################################################################
        """
        # Store the results, hyperparameters, and mean scores for this fold
        if not len(choose_hyperparam):
            name_hyperparam   = list(ml_algorithm.grid_result.best_params_.keys())
            choose_hyperparam = list(ml_algorithm.grid_result.best_params_.values())
            if chosen_algorithm == "MLP":
                choose_hyperparam[2] = str(choose_hyperparam[2]) # change tuple to string, otherwise does not work
        else:
            choose_hyperparam2 = list(ml_algorithm.grid_result.best_params_.values())
            if chosen_algorithm == "MLP":
                choose_hyperparam2[2] = str(choose_hyperparam2[2]) # change tuple to string, otherwise does not work
            choose_hyperparam = np.column_stack((choose_hyperparam, choose_hyperparam2))
            
        mean_scores_train.append(ml_algorithm.scores_train)
        mean_scores_test.append(ml_algorithm.scores_test)
        mean_MSE_scores_train.append(ml_algorithm.mse_scores_train)
        mean_MSE_scores_test.append(ml_algorithm.mse_scores_test)
        
        # Delete variables to save memory
        del X_train_val, y_train_val, X_test, y_test, train_ix, test_ix
        
        
        
    # Print results
    print("")
    print("")
    print("Chosen Algorithm:")
    print('mean_scores_train, R2:  %.2f (%.2f)' % (np.mean(mean_scores_train)*100, np.std(mean_scores_train)*100))
    print('mean_scores_test,  R2:  %.2f (%.2f)' % (np.mean(mean_scores_test)*100, np.std(mean_scores_test)*100))
    print('mean_scores_train, MSE: %.2f (%.2f)' % (np.mean(mean_MSE_scores_train), np.std(mean_MSE_scores_train)))
    print('mean_scores_test,  MSE: %.2f (%.2f)' % (np.mean(mean_MSE_scores_test), np.std(mean_MSE_scores_test)))
    
    print("")
    for i in range(len(name_hyperparam)):
        print("Hyperparameter " + name_hyperparam[i] + " = " + str(most_common(choose_hyperparam[i])))

    
    
    
    