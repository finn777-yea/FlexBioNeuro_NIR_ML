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
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.warn = warn
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)

class MLAlgorithm:
    def __init__(self, X_ml, y_ml, ps, n_jobs, show_plot, cal_case):
        """
        Initialize an MLAlgorithm object.

        Args:
            X_ml (numpy.ndarray): Input data for machine learning.
            y_ml (numpy.ndarray): Output data for machine learning.
            ps (PredefinedSplit): Predefined cross-validation split.
            n_jobs (int):         Number of CPU cores to use for parallel computation.
            show_plot (str):      A string indicating whether to display the plot. Should be 'yes' or 'no'.
        """
        self.X_ml         = X_ml
        self.y_ml         = y_ml
        self.groups       = None
        self.ps           = ps
        self.n_jobs       = n_jobs
        self.scores_train = None
        self.scores_test  = None
        self.show_plot    = show_plot
        self.cal_case     = cal_case
        
        
    def eval_gaussian_nb(self):
        """
        Evaluate Gaussian Naive Bayes algorithm with chosen hyperparameters.
        """
        if self.cal_case == "VFA_TA":
            gnb_all = GaussianNB(var_smoothing = 1e-5)
        elif self.cal_case == "Ac_acid":
            gnb_all = GaussianNB(var_smoothing = 1e-5)
            
        
        self.machine_learning_best_scores(gnb_all, "GaussianNB")
        self.plot_confusion_matrix(gnb_all, "GaussianNB")
        self.plot_learning_curve(gnb_all, "GaussianNB")
        self.plot_ROC_curves_and_AUC(gnb_all, "GaussianNB")
        
        
    def eval_ada_boost(self):
        """
        Evaluate Ada Boost Classifier algorithm with chosen hyperparameters.
        """
        if self.cal_case == "VFA_TA":
            abc_all = AdaBoostClassifier(learning_rate = 1.0,
                                         n_estimators  = 150,
                                         algorithm     = "SAMME.R",
                                         random_state  = 42)
        elif self.cal_case == "Ac_acid":
            abc_all = AdaBoostClassifier(learning_rate = 1.0,
                                         n_estimators  = 150,
                                         algorithm     = "SAMME.R",
                                         random_state  = 42)
        
        self.machine_learning_best_scores(abc_all, "AdaBoostClassifier")
        self.plot_confusion_matrix(abc_all, "AdaBoostClassifier")
        self.plot_learning_curve(abc_all, "AdaBoostClassifier")
        self.plot_ROC_curves_and_AUC(abc_all, "AdaBoostClassifier")
        
        
    def eval_knn(self):
        """
        Evaluate k-Nearest-Neighbors algorithm with chosen hyperparameters.
        """
        if self.cal_case == "VFA_TA":
            knn_all = KNeighborsClassifier(n_neighbors = 2,
                                           weights     = "distance",
                                           algorithm   = "auto",
                                           leaf_size   = 5,
                                           p           = 2)
        elif self.cal_case == "Ac_acid":
            knn_all = KNeighborsClassifier(n_neighbors = 4,
                                           weights     = "distance",
                                           algorithm   = "auto",
                                           leaf_size   = 5,
                                           p           = 2)
            
        self.machine_learning_best_scores(knn_all, "KNeighborsClassifier")
        self.plot_confusion_matrix(knn_all, "KNeighborsClassifier")
        self.plot_learning_curve(knn_all, "KNeighborsClassifier")
        self.plot_ROC_curves_and_AUC(knn_all, "KNeighborsClassifier")
        
        
    def eval_decision_tree(self):
        """
        Evaluate Decision Tree Classifier algorithm with chosen hyperparameters.
        """
        """
        # 10-Fold Cross Validation
        if self.cal_case == "VFA_TA":
            dtc_all = DecisionTreeClassifier(criterion         = "entropy",
                                             splitter          = "random",
                                             max_depth         = None,
                                             min_samples_split = 2,
                                             min_samples_leaf  = 1,
                                             max_features      = None,
                                             random_state      = 42)
        elif self.cal_case == "Ac_acid":
            dtc_all = DecisionTreeClassifier(criterion         = "gini",
                                             splitter          = "random",
                                             max_depth         = None,
                                             min_samples_split = 2,
                                             min_samples_leaf  = 1,
                                             max_features      = "log2",
                                             random_state      = 42)
        """ 
        
        # 5-Fold Cross Validation
        if self.cal_case == "VFA_TA":
            dtc_all = DecisionTreeClassifier(criterion         = "gini",
                                             splitter          = "best",
                                             max_depth         = None,
                                             min_samples_split = 7,
                                             min_samples_leaf  = 1,
                                             max_features      = None,
                                             random_state      = 42)
        elif self.cal_case == "Ac_acid":
            dtc_all = DecisionTreeClassifier(criterion         = "entropy",
                                             splitter          = "random",
                                             max_depth         = None,
                                             min_samples_split = 2,
                                             min_samples_leaf  = 3,
                                             max_features      = "sqrt",
                                             random_state      = 42)

        self.machine_learning_best_scores(dtc_all, "DecisionTreeClassifier")
        self.plot_confusion_matrix(dtc_all, "DecisionTreeClassifier")
        self.plot_learning_curve(dtc_all, "DecisionTreeClassifier")
        self.plot_ROC_curves_and_AUC(dtc_all, "DecisionTreeClassifier")
        
        
    def eval_extra_trees(self):
        """
        Evaluate Extra Trees Classifier algorithm with chosen hyperparameters.
        """
        if self.cal_case == "VFA_TA":
            etc_all = ExtraTreesClassifier(n_estimators      = 3,
                                           criterion         = "entropy",
                                           max_depth         = 10,
                                           min_samples_split = 5,
                                           min_samples_leaf  = 1,
                                           max_features      = None,
                                           bootstrap         = True,
                                           warm_start        = True,
                                           class_weight      = None,
                                           random_state      = 42)
        elif self.cal_case == "Ac_acid":
            etc_all = ExtraTreesClassifier(n_estimators      = 3,
                                           criterion         = "entropy",
                                           max_depth         = 10,
                                           min_samples_split = 2,
                                           min_samples_leaf  = 1,
                                           max_features      = "log2",
                                           bootstrap         = True,
                                           warm_start        = True,
                                           class_weight      = None,
                                           random_state      = 42)

        self.machine_learning_best_scores(etc_all, "ExtraTreesClassifier")
        self.plot_confusion_matrix(etc_all, "ExtraTreesClassifier")
        self.plot_learning_curve(etc_all, "ExtraTreesClassifier")
        self.plot_ROC_curves_and_AUC(etc_all, "ExtraTreesClassifier")
        
    def eval_random_forest(self):
        """
        Evaluate Random Forest Classifier algorithm with chosen hyperparameters.
        """
        
        # 10-Fold Cross Validation
        if self.cal_case == "VFA_TA":
            rfc_all = RandomForestClassifier(n_estimators      = 3,
                                             criterion         = "gini",
                                             max_depth         = None,
                                             min_samples_split = 2,
                                             min_samples_leaf  = 7,
                                             max_features      = "sqrt",
                                             bootstrap         = True,
                                             warm_start        = True,
                                             class_weight      = None,
                                             random_state      = 42) 
            
            
        elif self.cal_case == "Ac_acid":
            rfc_all = RandomForestClassifier(n_estimators      = 3,
                                             criterion         = "gini",
                                             max_depth         = 9,
                                             min_samples_split = 2,
                                             min_samples_leaf  = 1,
                                             max_features      = "sqrt",
                                             bootstrap         = True,
                                             warm_start        = True,
                                             class_weight      = None,
                                             random_state      = 42)
        
        """
        # 5-Fold Cross Validation
        if self.cal_case == "VFA_TA":
            rfc_all = RandomForestClassifier(n_estimators      = 3,
                                             criterion         = "entropy",
                                             max_depth         = 10,
                                             min_samples_split = 8,
                                             min_samples_leaf  = 1,
                                             max_features      = None,
                                             bootstrap         = True,
                                             warm_start        = True,
                                             class_weight      = None,
                                             random_state      = 50) 
            
        elif self.cal_case == "Ac_acid":
            rfc_all = RandomForestClassifier(n_estimators      = 3,
                                             criterion         = "gini",
                                             max_depth         = None,
                                             min_samples_split = 2,
                                             min_samples_leaf  = 1,
                                             max_features      = "log2",
                                             bootstrap         = True,
                                             warm_start        = True,
                                             class_weight      = None,
                                             random_state      = 10)
            
        """

        self.machine_learning_best_scores(rfc_all, "RandomForestClassifier")
        self.plot_confusion_matrix(rfc_all, "RandomForestClassifier")
        self.plot_learning_curve(rfc_all, "RandomForestClassifier")
        self.plot_ROC_curves_and_AUC(rfc_all, "RandomForestClassifier")
        
    def eval_grad_boosting(self):
        """
        Evaluate Gradient Boosting Classifier algorithm with chosen hyperparameters.
        """
        if self.cal_case == "VFA_TA":
            gbc_all = GradientBoostingClassifier(loss              = "exponential",
                                                 learning_rate     = 1.0,
                                                 n_estimators      = 3,
                                                 criterion         = "friedman_mse",
                                                 min_samples_split = 2,
                                                 min_samples_leaf  = 1,
                                                 max_depth         = None,
                                                 max_features      = "sqrt",
                                                 max_leaf_nodes    = None,
                                                 warm_start        = True,
                                                 tol               = 1e-3,
                                                 n_iter_no_change  = None,
                                                 random_state      = 42)
        elif self.cal_case == "Ac_acid":
            gbc_all = GradientBoostingClassifier(loss              = "exponential",
                                                 learning_rate     = 1.0,
                                                 n_estimators      = 3,
                                                 criterion         = "friedman_mse",
                                                 min_samples_split = 2,
                                                 min_samples_leaf  = 4,
                                                 max_depth         = None,
                                                 max_features      = "sqrt",
                                                 max_leaf_nodes    = None,
                                                 warm_start        = True,
                                                 tol               = 1e-3,
                                                 n_iter_no_change  = None,
                                                 random_state      = 42)

        self.machine_learning_best_scores(gbc_all, "GradientBoostingClassifier")
        self.plot_confusion_matrix(gbc_all, "GradientBoostingClassifier")
        self.plot_learning_curve(gbc_all, "GradientBoostingClassifier")
        self.plot_ROC_curves_and_AUC(gbc_all, "GradientBoostingClassifier")
        
        
    def eval_SVC(self):
        """
        Evaluate Support Vector Classifier algorithm with chosen hyperparameters.
        """
        if self.cal_case == "VFA_TA":
            svc_all = SVC(kernel       = "poly",
                          degree       = 2,
                          C            = 0.1,
                          tol          = 0.1,
                          gamma        = "scale",
                          probability  = True,
                          shrinking    = True,
                          class_weight = "balanced")
        elif self.cal_case == "Ac_acid":
            svc_all = SVC(kernel       = "poly",
                          degree       = 2,
                          C            = 0.1,
                          tol          = 0.1,
                          gamma        = "scale",
                          probability  = True,
                          shrinking    = True,
                          class_weight = "balanced")

        self.machine_learning_best_scores(svc_all, "SupportVectorClassifier")
        self.plot_confusion_matrix(svc_all, "SupportVectorClassifier")
        self.plot_learning_curve(svc_all, "SupportVectorClassifier")
        self.plot_ROC_curves_and_AUC(svc_all, "SupportVectorClassifier")
        
        
        
    def eval_MLP(self):
        """
        Evaluate Multiple Layer Perceptor Classifier algorithm with chosen hyperparameters.
        """
        if self.cal_case == "VFA_TA":
            mlp_all = MLPClassifier(hidden_layer_sizes = (10,),
                                    activation         = "relu",
                                    solver             = "lbfgs",
                                    learning_rate      = "constant",
                                    random_state       = 42,
                                    max_iter           = 5000,
                                    alpha              = 0.0001)
        elif self.cal_case == "Ac_acid":
            mlp_all = MLPClassifier(hidden_layer_sizes = (16,),
                                    activation         = "tanh",
                                    solver             = "sgd",
                                    learning_rate      = "constant",
                                    random_state       = 42,
                                    max_iter           = 5000,
                                    alpha              = 0.0001)
        

        self.machine_learning_best_scores(mlp_all, "MLPClassifier")
        self.plot_confusion_matrix(mlp_all, "MLPClassifier")
        self.plot_learning_curve(mlp_all, "MLPClassifier")
        self.plot_ROC_curves_and_AUC(mlp_all, "MLPClassifier")
        
        
    def machine_learning_best_scores(self, mod_best_param, model_name):
        """
        Calculate and print best scores for the selected machine learning model.

        Args:
            mod_best_param:   The best machine learning model obtained from hyperparameter tuning.
            model_name (str): The name of the machine learning model.
        """
        scores = cross_validate(mod_best_param, self.X_ml, self.y_ml, cv=self.ps, groups=self.groups,
                                scoring=('accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'),
                                return_train_score=True)

        # save all scores
        train1 = np.mean(scores['train_balanced_accuracy'])
        test1  = np.mean(scores['test_balanced_accuracy'])
        
        train2 = np.mean(scores['train_precision'])
        test2  = np.mean(scores['test_precision'])
        
        train3 = np.mean(scores['train_recall'])
        test3  = np.mean(scores['test_recall'])
        
        train4 = np.mean(scores['train_f1'])
        test4  = np.mean(scores['test_f1'])
        
        data      = [[f'{train1*100:.2f}%', f'{train2*100:.2f}%', f'{train3*100:.2f}%', f'{train4*100:.2f}%'],
                    [f'{test1*100:.2f}%', f'{test2*100:.2f}%', f'{test3*100:.2f}%', f'{test4*100:.2f}%']]
        index     = ["training-validation set", "test set"]
        columns   = ["balanced_accuracy", "precision", "recall", "f1-score"]
        df_scores = pd.DataFrame(data, index, columns)
        print(model_name)
        print(df_scores)
        print(' ')
        
    def plot_confusion_matrix(self, mod_best_param, model_name):
        """
        Plot the confusion matrix for a given model's predictions.
    
        This function generates a confusion matrix for a machine learning model's predictions and plots it as a heatmap.
    
        Args:
            mod_best_param: The trained machine learning model.
            model_name: A string representing the name of the model.
    
        Returns:
            None
        """
        X_train = self.X_ml[self.ps.test_fold == -1]
        y_train = self.y_ml[self.ps.test_fold == -1]
        X_test  = self.X_ml[self.ps.test_fold == 0]
        y_test  = self.y_ml[self.ps.test_fold == 0]
        
        mod_best_param.fit(X_train, y_train)
        y_pred_test = mod_best_param.predict(X_test)
        cf_matrix   = confusion_matrix(y_test, y_pred_test)
        
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
    
    def plot_learning_curve(self, mod_best_param, model_name, train_sizes=np.linspace(0.1, 1.0, 7)):
        """
        Plot the learning curve for a given model.
    
        This function generates a learning curve for a machine learning model and plots it to visualize the model's performance.
    
        Args:
            mod_best_param: The trained machine learning model.
            model_name: A string representing the name of the model.
            train_sizes: An array of training set sizes used to generate the learning curve.
    
        Returns:
            None
        """
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
    
            ax.set_title(model_name, fontsize = fontsize_title)
            
            ax.set_xlabel("Training examples", fontsize = fontsize_xlabel)
            ax.set_ylabel("Score", fontsize = fontsize_ylabel)
        
            train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(mod_best_param,
                                                                                  self.X_ml,
                                                                                  self.y_ml,
                                                                                  cv = self.ps,
                                                                                  n_jobs = self.n_jobs,
                                                                                  train_sizes = train_sizes,
                                                                                  return_times = True)
            
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std  = np.std(train_scores, axis=1)
            test_scores_mean  = np.mean(test_scores, axis=1)
            test_scores_std   = np.std(test_scores, axis=1)
        
            # Plot learning curve
            ax.grid(True, linewidth = grid_line_width, ls = '--')
            ax.fill_between(
                train_sizes,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.1,
                color="magenta",
            )
            ax.fill_between(
                train_sizes,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1,
                color="green",
            )
            ax.plot(
                train_sizes, train_scores_mean, "o-", color="magenta", label="Training score", markersize=4
            )
            ax.plot(
                train_sizes, test_scores_mean, "o-", color="green", label="Cross-validation score", markersize=4
            )
            ax.legend(loc="lower right", fontsize = fontsize_legend)
            
            # Set ylim
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.show()
    
    def plot_ROC_curves_and_AUC(self, mod_best_param, model_name):
        """
        Plot ROC curves and calculate the ROC AUC for a given classification model.
    
        This function generates ROC (Receiver Operating Characteristic) curves and calculates the ROC AUC (Area Under the Curve)
        to evaluate the performance of a classification model.
    
        Args:
            mod_best_param: The trained classification model.
            model_name: A string representing the name of the model.
    
        Returns:
            None
        """
        X_train = self.X_ml[self.ps.test_fold == -1]
        y_train = self.y_ml[self.ps.test_fold == -1]
        X_test  = self.X_ml[self.ps.test_fold == 0]
        y_test  = self.y_ml[self.ps.test_fold == 0]
        
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y_test))]
        
        # fit a model
        mod_best_param.fit(X_train, y_train)
        
        # predict probabilities
        lr_probs = mod_best_param.predict_proba(X_test)
        
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        
        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        
        # summarize scores
        print('Baseline: ROC AUC=%.3f' % (ns_auc))
        print('ML algorithm: ROC AUC=%.3f' % (lr_auc))
        
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        
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
    cal_case = "Ac_acid"
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
                                                      df_all             = data_processor.df_all)
        
    features_extractor.Scaling_data()
    features_extractor.extract_features()
    features_extractor.Predefined_cv()
        
        
        
        
        
    """
    ######################################################################################################
    # Run Machine Learning algrotihm, Class: MLAlgorithm
    ######################################################################################################
    """
    # Machine Learning Algorithm Setup
    ml_algorithm = MLAlgorithm(X_ml      = features_extractor.X_ml, 
                               y_ml      = features_extractor.y_ml, 
                               ps        = features_extractor.ps, 
                               n_jobs    = 6,
                               show_plot = "yes",
                               cal_case  = cal_case)
    
    # Initialize the dictionary to map algorithm names to functions
    algorithm_mapping = {
        "gaussian_nb":   ml_algorithm.eval_gaussian_nb,
        "ada_boost":     ml_algorithm.eval_ada_boost,
        "knn":           ml_algorithm.eval_knn,
        "decision_tree": ml_algorithm.eval_decision_tree,
        "extra_trees":   ml_algorithm.eval_extra_trees,
        "random_forest": ml_algorithm.eval_random_forest,
        "grad_boosting": ml_algorithm.eval_grad_boosting,
        "SVC":           ml_algorithm.eval_SVC,
        "MLP":           ml_algorithm.eval_MLP
        }
    
    # Specify the chosen algorithm (e.g., "decision_tree", "SVC", "random_forest")
    chosen_algorithm = "decision_tree"  # Change this to select the desired algorithm
    
    # Check if the chosen algorithm exists in the mapping, then call it
    if chosen_algorithm in algorithm_mapping:
        algorithm_mapping[chosen_algorithm]()
    else:
        print(f"Algorithm '{chosen_algorithm}' is not recognized.")

    print(" ")
    print(cal_case)
    
    
    