# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:42:03 2023

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
from sklearn.tree import DecisionTreeRegressor  # For Decision Trees regression
from sklearn.feature_selection import SelectKBest, f_regression  # For feature selection with chi-squared test
from sklearn.cross_decomposition import PLSRegression  # For Partial Least Squares regression
from scipy import signal  # For signal processing and detrending

# Machine learning regressors (Deep neural network)
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers

# Machine learning evaluation
from sklearn.model_selection import learning_curve      # learning curve
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


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


class SpectralPlotter:
    def __init__(self, show_plot="no"):
        """
        Initialize a SpectralPlotter object.

        Args:
            show_plot (str): A string indicating whether to display the plot. Should be 'yes' or 'no'.
        """
        self.show_plot = show_plot
        self.S1_4_WL   = range(1100,1352,2)
        self.S1_7_WL   = range(1350,1652,2)
        self.S2_0_WL   = range(1550,1952,2)
        self.S2_2_WL   = range(1750,2152,2)
    
    def plot_based_on_classes(self, all_WL, all_I, class_all, plot_label, title, xlabel, ylabel):
        """
        Plot data based on different classes.

        Args:
            all_WL (array-like):    Array of wavelength values.
            all_I (array-like):     Array of intensity values.
            class_all (array-like): Array of class labels for each data point.
            plot_label (list):      List of labels for the plot legend.
            title (str):            Title for the plot.
            xlabel (str):           Label for the x-axis.
            ylabel (str):           Label for the y-axis.

        Returns:
            fig (matplotlib.figure.Figure or None): The generated figure object if show_plot is 'yes', otherwise None.
        """
        
        # Edit the font and axes width
        mpl.rcParams['font.family']    = 'serif' # 'Avenir','serif'
        plt.rcParams['axes.linewidth'] = 1

        # Edit the font size and set the color of both classes
        font_size_title  = 6 + 2
        font_size_ylabel = 6 + 2
        font_size_xlabel = 6 + 2
        font_size_ticks  = 4 + 2
        font_size_legend = 4 + 2
        grid_line_width  = 0.5
        plot_line_width  = 0.5
        plot_marker      = 'None'
        plot_ls_0        = 'solid'
        plot_ls_1        = 'dashed'
        plot_marker_sz   = 1.0
        alpha_value      = 0.5

        pop_a = mpatches.Patch(color='green', label=plot_label[0])
        pop_b = mpatches.Patch(color='magenta', label=plot_label[1])
        
        # show_plot = 'yes' or 'no'
        if self.show_plot == "yes":
            # Create figure object and store it in a variable called 'fig'
            fig = plt.figure(figsize=(7.00, 3.50), dpi=300)

            for i in range(np.shape(all_I)[0]):
                if class_all[i] == 0:
                    plt.plot(all_WL,
                             all_I[i],
                             color='green',
                             linewidth=plot_line_width,
                             marker=plot_marker,
                             linestyle=plot_ls_0,
                             markersize=plot_marker_sz,
                             alpha=alpha_value)
                elif class_all[i] == 1:
                    plt.plot(all_WL,
                             all_I[i],
                             color='magenta',
                             linewidth=plot_line_width,
                             marker=plot_marker,
                             linestyle=plot_ls_1,
                             markersize=plot_marker_sz,
                             alpha=alpha_value)
                    
            # Set x-label, y-label and title
            plt.ylabel(ylabel, fontsize=font_size_ylabel)
            plt.xlabel(xlabel, fontsize=font_size_xlabel)
            plt.title(title, fontsize=font_size_title)
            
            # Set legend
            plt.legend(handles=[pop_a, pop_b], prop={'size': font_size_legend}, loc='upper right')

            # Hide the top and right spines of the axis
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            
            # Edit the major ticks of the x and y axes
            plt.gca().xaxis.set_tick_params(which='major', direction='in', top=False)
            plt.gca().yaxis.set_tick_params(which='major', direction='in', right=False)
            plt.tick_params(axis='both', which='major', labelsize=font_size_ticks)
            
            # Set the grid
            plt.gca().yaxis.grid(True, linewidth=grid_line_width, ls='dotted')
            
            # Set xticks
            plt.xticks([])

            end_S1_4_WL = len(self.S1_4_WL)
            end_S1_7_WL = len(self.S1_4_WL) + len(self.S1_7_WL)
            end_S2_0_WL = len(self.S1_4_WL) + len(self.S1_7_WL) + len(self.S2_0_WL)
            end_S2_2_WL = len(self.S1_4_WL) + len(self.S1_7_WL) + len(self.S2_0_WL) + len(self.S2_2_WL)

            x_ticks_pos = [0, int(len(self.S1_4_WL) / 2), end_S1_4_WL - 1,
                           end_S1_4_WL, end_S1_4_WL + int(len(self.S1_7_WL) / 2), end_S1_7_WL - 1,
                           end_S1_7_WL, end_S1_7_WL + int(len(self.S2_0_WL) / 2), end_S2_0_WL - 1,
                           end_S2_0_WL, end_S2_0_WL + int(len(self.S2_2_WL) / 2), end_S2_2_WL - 1]

            x_ticks_label = [str(self.S1_4_WL[0]), "...", str(self.S1_4_WL[-1]),
                             str(self.S1_7_WL[0]), "...", str(self.S1_7_WL[-1]),
                             str(self.S2_0_WL[0]), "...", str(self.S2_0_WL[-1]),
                             str(self.S2_2_WL[0]), "...", str(self.S2_2_WL[-1])]

            plt.xticks(x_ticks_pos, x_ticks_label)
            ticklabels = plt.gca().get_xticklabels()
            
            # set the x-ticks alignment
            ticklabels[0].set_ha("left")
            ticklabels[2].set_ha("right")
            ticklabels[3].set_ha("left")
            ticklabels[5].set_ha("right")
            ticklabels[6].set_ha("left")
            ticklabels[8].set_ha("right")
            ticklabels[9].set_ha("left")
            ticklabels[-1].set_ha("right")
            
            # show line, which separates the sensors
            y_min, y_max = plt.gca().get_ylim()
            plt.vlines(x=end_S1_4_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)
            plt.vlines(x=end_S1_7_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)
            plt.vlines(x=end_S2_0_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)
            
            # vertical shading to separate the sensors better
            plt.axvspan(end_S1_4_WL, end_S1_7_WL - 1, alpha=0.1)
            plt.axvspan(end_S2_0_WL, end_S2_2_WL - 1, alpha=0.1)

            fig.set_tight_layout(True)
            plt.show()
        else:
            fig = None
        return fig



class DataProcessor:
    def __init__(self, data_file_path, test_data_file_path, calibration_file_path, meas_sec):
        """
        Initialize a DataProcessor object.

        Args:
            data_file_path (str): Path to the data file.
            calibration_file_path (str): Path to the calibration file.
            meas_sec (int): Measurement duration in seconds.
        """
        self.data_file_path        = data_file_path
        self.test_data_file_path   = test_data_file_path
        self.calibration_file_path = calibration_file_path
        self.meas_sec              = meas_sec
        self.df_all                = None
        self.X_all                 = None
        self.y_all                 = None
        self.X_cal                 = None
        self.X_features            = None
        self.df_test               = None
        self.X_test                = None
        self.y_test                = None

    def load_data(self):
        """
        Load preprocessed variables from a CSV data file.
        """
        self.df_all = pd.read_csv(self.data_file_path, encoding='ISO-8859-1')
        
    def load_test_data(self):
        """
        Load preprocessed variables from a CSV data file.
        """
        self.df_test = pd.read_csv(self.test_data_file_path, encoding='ISO-8859-1')

    def set_XY_values(self):
        """
        Set X and Y values based on the loaded data.
        """
        self.X_all  = self.df_all.iloc[:, :-1].values
        self.y_all  = self.df_all.iloc[:, -1].values
        
        self.X_test = self.df_test.iloc[:, :-1].values
        self.y_test = self.df_test.iloc[:, -1].values

    def load_calibration(self):
        """
        Load calibration data and calculate absorbance (X_cal).
        """
        df_cal     = pd.read_csv(self.calibration_file_path, encoding='ISO-8859-1')
        cal_dark   = np.array(df_cal.iloc[0, :-1].values, dtype=np.float64)
        cal_ref    = np.array(df_cal.iloc[1, :-1].values, dtype=np.float64)
        self.X_cal  = -np.log10((self.X_all - cal_dark) / (cal_ref - cal_dark))
        self.X_test = -np.log10((self.X_test - cal_dark) / (cal_ref - cal_dark))
        
    def randomize(self):
        """
        Randomly shuffle the training data.
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
        self.y_all      = self.y_all[permutation_group]
        
        del permutation, permutation_group
        
        """
        Randomly shuffle the test data.
        """
        # Generate a random permutation of indices for data shuffling
        permutation = np.random.permutation(int(self.y_test.shape[0] / self.meas_sec))
        
        # Create a permutation group to apply to both X and Y
        permutation_group = np.repeat(permutation * 0, self.meas_sec)
        
        # Populate the permutation group with corresponding indices
        for i in range(len(permutation)):
            permutation_group[i * self.meas_sec:(i + 1) * self.meas_sec] = np.arange(permutation[i] * self.meas_sec, (permutation[i] + 1) * self.meas_sec)
        
        # Shuffle the data based on the permutation
        self.X_test = self.X_test[permutation_group, :]
        self.y_test = self.y_test[permutation_group]
        
        del permutation, permutation_group
        
        
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
        Apply baseline correction to training data.

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
        
        del X_S1_4, X_S1_7, X_S2_0, X_S2_2
        
        """
        Apply baseline correction to test data.

        Args:
            method (int): The method to use for baseline correction.
        """
        X_S1_4, X_S1_7, X_S2_0, X_S2_2 = self.split_blocks(copy.deepcopy(self.X_test))
        
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
        self.X_test = np.hstack((X_S1_4, X_S1_7, X_S2_0, X_S2_2))
        
        del X_S1_4, X_S1_7, X_S2_0, X_S2_2
        


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
        Perform feature reduction using DecisionTreeRegressor.

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            y_train_val (numpy.ndarray): Training and validation output data.
            df_all (pd.DataFrame):       Dataframe containing all features.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data and the selected feature names.
        """
        # Feature reduction using DecisionTreeRegressor
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
        # Univariate feature selection using f_regression test
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
    def __init__(self, X_ml, y_ml, ps, meas_sec, epochs=200, show_plot="no"):
        """
        Initialize the NeuralNetworkOptimizer class.

        Args:
            X_ml (numpy.ndarray): Input features for machine learning.
            y_ml (numpy.ndarray): Target labels for machine learning.
            ps (PredefinedSplit): Predefined split object for cross-validation.
            meas_sec (int): Measurement section value.
            epochs (int): Number of training epochs for neural networks (default is 200).
            show_plot (str): Whether to display plots ("yes" or "no").

        Attributes:
            X_ml (numpy.ndarray): Input features for machine learning.
            y_ml (numpy.ndarray): Target labels for machine learning.
            ps (PredefinedSplit): Predefined split object for cross-validation.
            meas_sec (int): Measurement section value.
            epochs (int): Number of training epochs for neural networks.
            show_plot (str): Whether to display plots ("yes" or "no").
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.
        """
        self.X_ml      = X_ml
        self.y_ml      = y_ml
        self.ps        = ps
        self.meas_sec  = meas_sec
        self.epochs    = epochs
        self.show_plot = show_plot
        self.X_train   = self.X_ml[self.ps.test_fold == -1]
        self.y_train   = self.y_ml[self.ps.test_fold == -1]
        self.X_test    = self.X_ml[self.ps.test_fold == 0]
        self.y_test    = self.y_ml[self.ps.test_fold == 0]
        
        
    def create_baseline(self):
        """
        Create a baseline deep neural network model.

        Returns:
            keras.models.Sequential: Compiled Keras model for the deep neural network.
        """
        # generate model
        model = Sequential()
        model.add(Dense(units                = 14, 
                        input_dim            = np.shape(self.X_train)[1], 
                        activation           = "tanh", 
                        kernel_initializer   = initializers.HeUniform(seed=42), 
                        bias_initializer     = initializers.LecunNormal(seed=42),
                        kernel_regularizer   = "L2",
                        bias_regularizer     = "L1",
                        activity_regularizer = "L1",
                        kernel_constraint    = "MaxNorm",
                        bias_constraint      = "UnitNorm"))
        
        model.add(Dense(units                = 8, 
                        activation           = "relu", 
                        kernel_initializer   = initializers.HeUniform(seed=42), 
                        bias_initializer     = initializers.LecunNormal(seed=42),
                        kernel_regularizer   = "L2",
                        bias_regularizer     = "L1",
                        activity_regularizer = "L1",
                        kernel_constraint    = "MaxNorm",
                        bias_constraint      = "UnitNorm"))
    
        model.add(Dense(units                = 1,
                        activation           = "relu", 
                        kernel_initializer   = initializers.HeUniform(seed=42), 
                        bias_initializer     = initializers.LecunNormal(seed=42),
                        kernel_regularizer   = "L2",
                        bias_regularizer     = "L1",
                        activity_regularizer = "L1",
                        kernel_constraint    = "MaxNorm",
                        bias_constraint      = "UnitNorm"))
        
        # compile model
        model.compile(loss='mean_squared_error', optimizer="Nadam", metrics=['mean_absolute_error'])
        return model
    
    def evaluate_nn(self):
        """
        Train and evaluate the deep neural network model.

        This method trains the deep neural network, calculates evaluation scores, and displays plots.

        Returns:
            None
        """
        estimator = self.create_baseline()
        estimator.fit(x               = self.X_train, 
                      y               = self.y_train, 
                      epochs          = self.epochs, 
                      batch_size      = 32,
                      verbose         = 0,
                      validation_data = (self.X_test, self.y_test))
        
        self.machine_learning_best_scores(estimator, "Deep Neural Network")
        self.plot_learning_curve(self.create_baseline(), "Deep Neural Network")
        self.plot_best_results(estimator, "Deep Neural Network")
        self.plot_regression_result(estimator, "Deep Neural Network")
        
        
    def machine_learning_best_scores(self, estimator, model_name):
        """
        Calculate and display machine learning evaluation scores.

        Args:
            estimator (keras.models.Sequential): Trained deep neural network model.
            model_name (str): Name of the machine learning model.

        Returns:
            None
        """
        preds_train = estimator.predict(self.X_train)
        preds_test  = estimator.predict(self.X_test)
            
        train1      = r2_score(self.y_train,preds_train)
        test1       = r2_score(self.y_test,preds_test)
        
        train2      = mean_squared_error(self.y_train,preds_train,squared=False)
        test2       = mean_squared_error(self.y_test,preds_test,squared=False)
    
        train3      = mean_absolute_error(self.y_train,preds_train)
        test3       = mean_absolute_error(self.y_test,preds_test)
        
        train4      = explained_variance_score(self.y_train,preds_train)
        test4       = explained_variance_score(self.y_test,preds_test)
        
        data      = [[f'{train1*100:.2f}%', f'{train2:.2f}', f'{train3:.2f}', f'{train4*100:.2f}%'],
                    [f'{test1*100:.2f}%', f'{test2:.2f}', f'{test3:.2f}', f'{test4*100:.2f}%']]
        index     = ["training-validation set", "test set"]
        columns   = ["r2", "RMSE", "MAE", "explained_variance"]
        df_scores = pd.DataFrame(data, index, columns)
        print(model_name)
        print(df_scores)
        print(' ')
        
    
    def plot_learning_curve(self, estimator, model_name):
        """
        Plot the learning curve for the training and validation sets.

        Args:
            estimator (keras.models.Sequential): Trained deep neural network model.
            model_name (str): Name of the machine learning model.

        Returns:
            None
        """
        # Check if the plot should be displayed
        if self.show_plot == "yes":
            
            fontsize_title  = 10
            fontsize_xlabel = 8
            fontsize_ylabel = 8
            fontsize_xtick  = 6
            fontsize_ytick  = 6
            fontsize_legend = 6
            grid_line_width = 0.5
            
            plt.rc('font', family = 'serif')
            plt.rc('xtick', labelsize = fontsize_xtick)
            plt.rc('ytick', labelsize = fontsize_ytick)
            plt.rcParams['figure.dpi'] = 450
            plt.rcParams["figure.autolayout"] = True
            
            history = estimator.fit(x               = self.X_train, 
                                    y               = self.y_train, 
                                    epochs          = self.epochs, 
                                    batch_size      = 32,
                                    verbose         = 0,
                                    validation_data = (self.X_test, self.y_test))
        
    
            history_dict    = history.history
            loss_values     = history_dict['loss']
            val_loss_values = history_dict['val_loss']
            accuracy        = history_dict['mean_absolute_error']
            val_accuracy    = history_dict['val_mean_absolute_error']
            epochs          = range(1, len(loss_values) + 1)
            
            fig = plt.figure(figsize=(4.00,3.00))
            ax = fig.subplots()
            
            # Hide the top and right spines of the axis
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Edit the major and minor ticks of the x and y axes
            ax.xaxis.set_tick_params(which='major', size=5, width=1.0, direction='in', top=False)
            ax.yaxis.set_tick_params(which='major', size=5, width=1.0, direction='in', right=False)
    
            ax.set_title(model_name, fontsize = fontsize_title)
            
            ax.set_xlabel("Epochs", fontsize = fontsize_xlabel)
            ax.set_ylabel("mean_absolute_error", fontsize = fontsize_ylabel)
            
            # Plot learning curve
            ax.grid(True, linewidth = grid_line_width, ls = '--')
            ax.plot(
                epochs, accuracy, color="magenta", label="Training MAE")
            ax.plot(
                epochs, val_accuracy, color="green", label="Validation MAE")
            ax.legend(loc="lower right", fontsize = fontsize_legend)
            
            # Set ylim
            ax.set_ylim(0, 5)
            #y_lim_min = np.min([np.min(accuracy), np.min(val_accuracy)])
            #y_lim_max = np.max([np.max(accuracy), np.max(val_accuracy)])
            #ax.set_ylim(y_lim_min, y_lim_max)

            plt.tight_layout()
            plt.show()
    
    
    def plot_best_results(self, mod_best_param, model_name):
        """
        Plot the best results of a machine learning model and calculate RMSE.
    
        Args:
            mod_best_param (estimator): The best-tuned machine learning model.
            model_name (str): The name of the machine learning model.
    
        Returns:
            None: The function displays the plot but doesn't return a value.
        """
        # Fit the model on the training data and make predictions on the test data
        best_y_pred_test = mod_best_param.predict(self.X_test)
        best_y_test      = self.y_test
        
        best_y_pred_test = np.mean(best_y_pred_test.reshape(-1, self.meas_sec), axis=1)
        best_y_test      = np.mean(best_y_test.reshape(-1, self.meas_sec), axis=1)
    
        RMSE_values      = mean_squared_error(best_y_test, best_y_pred_test, squared=False)
        title = model_name + ": Plot test data, RMSE = " + "{:.1f}".format(RMSE_values) + "%"
        
        if self.show_plot == "yes":
            
            fontsize_title  = 10
            fontsize_xlabel = 8
            fontsize_ylabel = 8
            fontsize_xtick  = 6
            fontsize_ytick  = 6
            fontsize_legend = 6
            grid_line_width = 0.5
            
            plt.rc('font', family = 'serif')
            plt.rc('xtick', labelsize = fontsize_xtick)
            plt.rc('ytick', labelsize = fontsize_ytick)
            plt.rcParams['figure.dpi'] = 450
            plt.rcParams["figure.autolayout"] = True
            
            fig = plt.figure(figsize=(4.00,3.00))
            ax = fig.subplots()
            
            # Hide the top and right spines of the axis
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Edit the major and minor ticks of the x and y axes
            ax.xaxis.set_tick_params(which='major', size=5, width=1.0, direction='in', top=False)
            ax.yaxis.set_tick_params(which='major', size=5, width=1.0, direction='in', right=False)
    
            ax.set_title(title, fontsize = fontsize_title)
            ax.set_ylim(np.min([np.min(best_y_test),np.min(best_y_pred_test)]), 
                        np.max([np.max(best_y_test),np.max(best_y_pred_test)]))
            ax.set_xlabel("Samples", fontsize = fontsize_xlabel)
            ax.set_ylabel("Dry Matter content [%]", fontsize = fontsize_ylabel)
        
            # Plot
            x_axis = np.linspace(1,np.size(best_y_test),np.size(best_y_test))
            ax.grid(True, linewidth = grid_line_width, ls = '--')
            ax.plot(x_axis, 100-best_y_test, "o", color="magenta", label="Real values", markersize=4)
            ax.plot(x_axis, 100-best_y_pred_test, "s", color="green", label="Prediction values", markersize=4, alpha=0.5)
            #ax.set_ylim(80, 98)
            ax.set_ylim(100-98, 100-80)
            ax.legend(loc="best", fontsize = fontsize_legend)
            
            plt.tight_layout()
            plt.show()
            
            
    def plot_regression_result(self, mod_best_param, model_name):
        """
        Plot the regression results of a machine learning model and calculate R^2.
    
        Parameters:
            mod_best_param (estimator): The best-tuned machine learning model.
            model_name (str): The name of the machine learning model.
    
        Returns:
            None: The function displays the plot but doesn't return a value.
        """
        # Fit the model on the training data and make predictions on the test data
        best_y_pred_test = mod_best_param.predict(self.X_test)
        best_y_test      = self.y_test
        
        # Calculate mean from 8 measurement seconds
        best_y_pred_test = np.mean(best_y_pred_test.reshape(-1, self.meas_sec), axis=1)
        best_y_test      = np.mean(best_y_test.reshape(-1, self.meas_sec), axis=1)
        
        # Change moisture to DM content
        best_y_pred_test = 100-best_y_pred_test
        best_y_test      = 100-best_y_test
        
        # Calculate R2 values
        R2_score         = r2_score(best_y_test, best_y_pred_test)
        
        # Calculate regression line
        m, b = np.polyfit(best_y_pred_test, best_y_test, 1)
        
        title = model_name + ": Plot regression result"
        
        if self.show_plot == "yes":
            # Define font sizes and grid line width for the plot
            fontsize_title  = 10
            fontsize_xlabel = 8
            fontsize_ylabel = 8
            fontsize_xtick  = 6
            fontsize_ytick  = 6
            fontsize_legend = 6
            grid_line_width = 0.5
            plot_line_width = 2
            
            # Configure plot font settings
            plt.rc('font', family = 'serif')
            plt.rc('xtick', labelsize = fontsize_xtick)
            plt.rc('ytick', labelsize = fontsize_ytick)
            plt.rcParams['figure.dpi'] = 450
            plt.rcParams["figure.autolayout"] = True
            
            # Create a figure and axis for the plot
            fig = plt.figure(figsize=(4.00,3.00))
            ax = fig.subplots()
            
            # Hide the top and right spines of the axis
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Edit the major and minor ticks of the x and y axes
            ax.xaxis.set_tick_params(which='major', size=5, width=1.0, direction='in', top=False)
            ax.yaxis.set_tick_params(which='major', size=5, width=1.0, direction='in', right=False)
            
            # Set the title and labels for the plot
            ax.set_title(title, fontsize = fontsize_title)
            ax.set_ylim(np.min([np.min(best_y_test),np.min(best_y_pred_test)]), 
                        np.max([np.max(best_y_test),np.max(best_y_pred_test)]))
            ax.set_xlim(np.min([np.min(best_y_test),np.min(best_y_pred_test)]), 
                        np.max([np.max(best_y_test),np.max(best_y_pred_test)]))
            ax.set_xlabel("Estimated [g/l]", fontsize = fontsize_xlabel)
            ax.set_ylabel("Measured [g/l]", fontsize = fontsize_ylabel)
        
            # Plot the real and predicted values
            ax.grid(True, linewidth = grid_line_width, ls = '--')
            ax.plot(best_y_pred_test, best_y_test, "o", color="magenta", label=" ", markersize=4)
            ax.plot(best_y_pred_test, m*best_y_pred_test + b, color="green", linewidth = plot_line_width)
            text_r2 = "$\mathregular{r^2}$ = " + "{:.1f}".format(R2_score*100) + "%"
            ax.text(14, 6, text_r2, color = 'grey')
            #ax.legend(loc="lower left", fontsize = fontsize_legend)
            
            # Display the plot
            plt.tight_layout()
            plt.show()




#%% Main
if __name__ == "__main__":
    """
    ######################################################################################################
    # Processing data, Class: DataProcessor
    ######################################################################################################
    """
    # Initialize Data Processor
    data_processor = DataProcessor(data_file_path        = data_path("regression_NIR_Data_raw_DM.csv"),
                                   test_data_file_path   = data_path("Test_regression_NIR_Data_raw_DM.csv"),
                                   calibration_file_path = calibration_path(),
                                   meas_sec              = 8)

    # Load raw data, set up features and targets, load calibration data, and perform baseline correction
    data_processor.load_data()
    data_processor.load_test_data()
    data_processor.set_XY_values()
    data_processor.load_calibration()
    data_processor.baseline_correction(method=2)
    data_processor.randomize()
    
    
    
    
    
    """
    ######################################################################################################
    # Extract features, Class: Scaling_and_FeatureExtractor
    ######################################################################################################
    """
    # Feature Extraction and Predefined Cross-Validation Setup
    features_extractor = Scaling_and_FeatureExtractor(number_of_features = 5, 
                                                      X_train_val        = data_processor.X_features, 
                                                      y_train_val        = data_processor.y_all, 
                                                      X_test             = data_processor.X_test, 
                                                      y_test             = data_processor.y_test, 
                                                      df_all             = data_processor.df_all)
        
    features_extractor.Scaling_data()
    features_extractor.extract_features()
    features_extractor.Predefined_cv()
    
    
    
    
    
    """
    ######################################################################################################
    # Evaluate Machine Learning, Class: NeuralNetworkOptimizer
    ######################################################################################################
    """
    # Machine Learning Algorithm Setup
    nn_optimizer = NeuralNetworkOptimizer(X_ml      = features_extractor.X_ml, 
                                          y_ml      = features_extractor.y_ml, 
                                          ps        = features_extractor.ps, 
                                          meas_sec  = data_processor.meas_sec,
                                          epochs    = 500,
                                          show_plot = "yes")
    
    # Evaluate the chosen neural network
    nn_optimizer.evaluate_nn()
    


