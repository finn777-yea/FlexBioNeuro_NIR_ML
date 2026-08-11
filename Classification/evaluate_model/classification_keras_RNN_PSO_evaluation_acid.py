# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:45:51 2023

@author: ge23hum
"""


from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.helpers import warn
from common.data import DataProcessor
from common.features import Scaling_and_FeatureExtractor

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import initializers
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.utils import set_random_seed

warnings.warn = warn
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
set_random_seed(42)

class NeuralNetworkOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, meas_sec, cal_case, epochs=1000, show_plot="no"):
        """
        Initialize the NeuralNetworkOptimizer class.

        Args:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.
            meas_sec (int): Measurement section value.
            epochs (int): Number of training epochs for neural networks (default is 1000).
            show_plot (str): Whether to display plots ("yes" or "no").

        Attributes:
            meas_sec (int): Measurement section value.
            epochs (int): Number of training epochs for neural networks.
            show_plot (str): Whether to display plots ("yes" or "no").
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.
        """
        self.meas_sec  = meas_sec
        self.epochs    = epochs
        self.show_plot = show_plot
        self.cal_case  = cal_case
        self.X_train   = X_train
        self.y_train   = y_train
        self.X_test    = X_test
        self.y_test    = y_test
        
        
    def create_baseline(self):
        """
        Create a baseline deep neural network model.

        Returns:
            keras.models.Sequential: Compiled Keras model for the deep neural network.
        """
        if self.cal_case == "VFA_TA":
            # generate model
            model = Sequential()
            model.add(LSTM(units                 = 7, 
                           input_shape           = (self.meas_sec, self.X_train.shape[2]), 
                           activation            = "softplus",
                           recurrent_activation  = "softsign",
                           kernel_initializer    = initializers.GlorotUniform(seed=42),
                           recurrent_initializer = initializers.GlorotUniform(seed=42),
                           bias_initializer      = initializers.VarianceScaling(seed=42),
                           kernel_regularizer    = "L2",
                           bias_regularizer      = "L2",
                           activity_regularizer  = "L1L2",
                           kernel_constraint     = "UnitNorm",
                           bias_constraint       = "UnitNorm",
                           return_sequences      = False))
        
            model.add(Dense(units                = 1,
                            activation           = "sigmoid", 
                            kernel_initializer   = initializers.GlorotUniform(seed=42),
                            bias_initializer     = initializers.VarianceScaling(seed=42),
                            kernel_regularizer   = "L2",
                            bias_regularizer     = "L2",
                            activity_regularizer = "L1L2",
                            kernel_constraint    = "UnitNorm",
                            bias_constraint      = "UnitNorm"))
            
            # compile model
            model.compile(loss='binary_crossentropy', optimizer="Nadam", metrics=['binary_accuracy'])
            
        elif self.cal_case == "Ac_acid":
            # generate model
            model = Sequential()
            model.add(LSTM(units                 = 9, 
                           input_shape           = (self.meas_sec, self.X_train.shape[2]), 
                           activation            = "softsign",
                           recurrent_activation  = "tanh",
                           kernel_initializer    = initializers.LecunUniform(seed=42),
                           recurrent_initializer = initializers.LecunUniform(seed=42),
                           bias_initializer      = initializers.GlorotNormal(seed=42),
                           kernel_regularizer    = "L1L2",
                           bias_regularizer      = "L1L2",
                           activity_regularizer  = "L1L2",
                           kernel_constraint     = "NonNeg",
                           bias_constraint       = "NonNeg",
                           return_sequences      = False))
        
            model.add(Dense(units                = 1,
                            activation           = "sigmoid", 
                            kernel_initializer   = initializers.LecunUniform(seed=42),
                            bias_initializer     = initializers.GlorotNormal(seed=42),
                            kernel_regularizer   = "L1L2",
                            bias_regularizer     = "L1L2",
                            activity_regularizer = "L1L2",
                            kernel_constraint    = "NonNeg",
                            bias_constraint      = "NonNeg"))
            
            # compile model
            model.compile(loss='binary_crossentropy', optimizer="RMSprop", metrics=['binary_accuracy'])
        
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
        self.plot_confusion_matrix(estimator, "Deep Neural Network")
        self.plot_learning_curve(self.create_baseline(), "Deep Neural Network")
        self.plot_ROC_curves_and_AUC(estimator, "Deep Neural Network")
        
        
        
    def machine_learning_best_scores(self, estimator, model_name):
        """
        Calculate and display machine learning evaluation scores.

        Args:
            estimator (keras.models.Sequential): Trained deep neural network model.
            model_name (str): Name of the machine learning model.

        Returns:
            None
        """
        preds_train = (estimator.predict(self.X_train) > 0.5).astype("int32")
        preds_test  = (estimator.predict(self.X_test) > 0.5).astype("int32")
            
        train1      = balanced_accuracy_score(self.y_train[:,0],preds_train[:,0])
        test1       = balanced_accuracy_score(self.y_test[:,0],preds_test[:,0])
        
        train2      = precision_score(self.y_train[:,0],preds_train[:,0])
        test2       = precision_score(self.y_test[:,0],preds_test[:,0])
    
        train3      = recall_score(self.y_train[:,0],preds_train[:,0])
        test3       = recall_score(self.y_test[:,0],preds_test[:,0])
        
        train4      = f1_score(self.y_train[:,0],preds_train[:,0])
        test4       = f1_score(self.y_test[:,0],preds_test[:,0])
        
        data      = [[f'{train1*100:.2f}%', f'{train2*100:.2f}%', f'{train3*100:.2f}%', f'{train4*100:.2f}%'],
                    [f'{test1*100:.2f}%', f'{test2*100:.2f}%', f'{test3*100:.2f}%', f'{test4*100:.2f}%']]
        index     = ["training-validation set", "test set"]
        columns   = ["balanced_accuracy", "precision", "recall", "f1-score"]
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
            accuracy        = history_dict['binary_accuracy']
            val_accuracy    = history_dict['val_binary_accuracy']
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
            ax.set_ylabel("Score", fontsize = fontsize_ylabel)
            
            # Plot learning curve
            ax.grid(True, linewidth = grid_line_width, ls = '--')
            ax.plot(
                epochs, accuracy, color="magenta", label="Training accuracy")
            ax.plot(
                epochs, val_accuracy, color="green", label="Validation accuracy")
            ax.legend(loc="lower right", fontsize = fontsize_legend)
            
            # Set ylim
            ax.set_ylim(0, 1)
            
            
            plt.tight_layout()
            plt.show()
    
    
    def plot_confusion_matrix(self, mod_best_param, model_name):
        """
        Plot a confusion matrix for a binary classification model and visualize the predicted versus actual outcomes.
    
        This function generates a confusion matrix and visualizes it as a heatmap to assess the performance of a binary classification model.
    
        Args:
            mod_best_param: The trained binary classification model.
            model_name: A string representing the name of the model.
    
        Returns:
            None
        """
        y_pred_test = (mod_best_param.predict(self.X_test) > 0.5).astype("int32")
        cf_matrix   = confusion_matrix(self.y_test[:,0], y_pred_test[:,0])
        
        if self.show_plot == "yes":
            plt.figure(figsize=(4.00,3.00))
            
            #cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
            cmn = cf_matrix.astype('int')
            
            #ax = sns.heatmap(cmn*100, annot=True, fmt='.1f', vmin=0, vmax=100, cmap="Blues")
            ax = sns.heatmap(cmn, annot=True, vmin=0, vmax=cf_matrix.max(), cmap="YlGnBu")
            
            ax.set_title(model_name);
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ');
            
            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(['Class 0','Class 1'])
            ax.yaxis.set_ticklabels(['Class 0','Class 1'])
            
            ## Display the visualization of the Confusion Matrix.
            plt.tight_layout()
            plt.show()
            
    def plot_ROC_curves_and_AUC(self, mod_best_param, model_name):
        """
        Plots ROC curves and calculates the AUC (Area Under the Curve) scores for a binary classification model.
    
        Args:
            mod_best_param (object): The best-performing binary classification model.
            model_name (str): The name of the model for labeling the ROC curve plot.
    
        Returns:
            None
    
        This function generates a "no skill" prediction (majority class) as a baseline and compares it with the provided model's predictions.
        It calculates the ROC AUC scores for both models and, if specified, displays a ROC curve plot.
        """
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(self.y_test))]
        
        # predict probabilities
        lr_probs = mod_best_param.predict(self.X_test)
        
        # calculate scores
        ns_auc = roc_auc_score(self.y_test, ns_probs)
        lr_auc = roc_auc_score(self.y_test, lr_probs[:,0])
        
        # summarize scores
        print('Baseline: ROC AUC=%.3f' % (ns_auc))
        print('ML algorithm: ROC AUC=%.3f' % (lr_auc))
        
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, lr_probs)
        
        if self.show_plot == "yes":
            plt.figure(figsize=(4.00,3.00))
        
            # plot the roc curve for the model
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
            
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(model_name)
            
            # show the legend
            plt.legend()
            
            # show the plot
            plt.show()





#%% Main
if __name__ == "__main__":
    """
    ######################################################################################################
    # Calibration case: VFA/TA-ratio or acetic acid concentration
    ######################################################################################################
    """
    cal_case = "VFA_TA"
    if cal_case == "VFA_TA":
        data_file_path      = "Data/classification_NIR_Data_raw_VFA_TA.csv"
        test_data_file_path = "Data/Test_classification_NIR_Data_raw_VFA_TA.csv"
        calibrate           = "no"
    elif cal_case == "Ac_acid":
        data_file_path      = "Data/classification_NIR_Data_raw_Ac_acid.csv"
        test_data_file_path = "Data/Test_classification_NIR_Data_raw_Ac_acid.csv"
        calibrate           = "yes"
    
    
    
    
    
    """
    ######################################################################################################
    # Processing data, Class: DataProcessor
    ######################################################################################################
    """
    # Initialize Data Processor
    data_processor = DataProcessor(data_file_path        = data_file_path,
                                   test_data_file_path   = test_data_file_path,
                                   calibration_file_path = "Data/calibration_NIR_Data.csv",
                                   meas_sec              = 8)

    # Load raw data, set up features and targets, load calibration data, and perform baseline correction
    data_processor.load_data()
    data_processor.load_test_data()
    data_processor.set_XY_values()
    data_processor.load_calibration(calibrate = calibrate)
    data_processor.randomize()
    X_train_val_a, X_test_a = data_processor.baseline_correction(method=2)
    X_train_val_b, X_test_b = data_processor.baseline_correction(method=3)
    X_train_val_c, X_test_c = data_processor.baseline_correction(method=4)
    
        
        
        
        
        
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
                                                      y_train_val        = data_processor.y_all, 
                                                      X_test_a           = X_test_a, 
                                                      X_test_b           = X_test_b, 
                                                      X_test_c           = X_test_c, 
                                                      y_test             = data_processor.y_test, 
                                                      df_all             = data_processor.df_all,
                                                      meas_sec           = data_processor.meas_sec)
        
    features_extractor.Scaling_data()
    features_extractor.extract_features()
    features_extractor.reshape_data()
        
        
        
        
        
    """
    ######################################################################################################
    # Evaluate Machine Learning, Class: NeuralNetworkOptimizer
    ######################################################################################################
    """
    # Machine Learning Algorithm Setup
    nn_optimizer = NeuralNetworkOptimizer(X_train   = features_extractor.X_out, 
                                          y_train   = features_extractor.y_train_val, 
                                          X_test    = features_extractor.X_out_test, 
                                          y_test    = features_extractor.y_test, 
                                          meas_sec  = data_processor.meas_sec,
                                          epochs    = 500,
                                          show_plot = "yes",
                                          cal_case  = cal_case)
    
    # Evaluate the chosen neural network
    nn_optimizer.evaluate_nn()

    
    
    
    