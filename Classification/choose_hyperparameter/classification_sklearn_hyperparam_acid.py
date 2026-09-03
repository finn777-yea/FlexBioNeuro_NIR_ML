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

import random
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.warn = warn
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)

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
        self.X_ml         = X_ml
        self.y_ml         = y_ml
        self.groups       = None
        self.ps           = ps
        self.n_jobs       = n_jobs
        self.grid_result  = None
        self.scores_train = None
        self.scores_test  = None

    def run_knn(self):
        """
        Run k-Nearest Neighbors (k-NN) algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_neighbors" : [2, 3, 4, 5, 6, 7],
            "weights"     : ["uniform", "distance"],
            "algorithm"   : ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size"   : [5, 10, 15],
            "p"           : [1, 2]
            }
        
        self.machine_learning_grid_search(KNeighborsClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "kNN")

    def run_gaussian_nb(self):
        """
        Run Gaussian Naive Bayes (GNB) algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "var_smoothing" : [1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            }

        self.machine_learning_grid_search(GaussianNB(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "GNB")

    def run_bernoulli_nb(self):
        """
        Run Bernoulli Naive Bayes algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "alpha"       : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "force_alpha" : [True, False],
            "binarize"    : [0.0, None],
            "fit_prior"   : [True, False]
            }

        self.machine_learning_grid_search(BernoulliNB(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "BernoulliNB")
        
    def run_decision_tree(self):
        """
        Run Decision Tree Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "criterion"         : ["gini", "entropy", "log_loss"],
            "splitter"          : ["best", "random"],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
        	"min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
        	"max_features"      : [None, "sqrt", "log2"],
            "random_state"      : [42]
            }

        self.machine_learning_grid_search(DecisionTreeClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "DecisionTreeClassifier")
    
    def run_lda(self):
        """
        Run Linear Discriminant Analysis algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "solver"           : ["svd", "lsqr", "eigen"],
            "shrinkage"        : ["auto", None],
            "store_covariance" : [True, False],
        	"tol"              : [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
            }

        self.machine_learning_grid_search(LinearDiscriminantAnalysis(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "LinearDiscriminantAnalysis")
        
    def run_logreg(self):
        """
        Run Logistic Regression algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "penalty"    : ["l1", "l2", "elasticnet"],
            "dual"       : [True, False],
            "C"          : [0.01, 0.1, 1.0, 10.0],
        	"tol"        : [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5],
        	"solver"     : ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        	"max_iter"   : [5000],
        	"warm_start" : [True, False],
            }

        self.machine_learning_grid_search(LogisticRegression(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "LogisticRegression")
        
    def run_SVC(self):
        """
        Run Support Vector Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "kernel"       : ["linear", "poly", "rbf", "sigmoid"],
            "degree"       : [2, 3, 4],
            "C"            : [0.001, 0.01, 0.1, 1.0, 10.0],
        	"tol"          : [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4],
        	"gamma"        : ["scale", "auto"],
        	"probability"  : [True, False],
        	"shrinking"    : [True, False],
        	"class_weight" : ["balanced", None]
            }

        self.machine_learning_grid_search(SVC(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "SVC")
        
    def run_random_forest(self):
        """
        Run Random Forest Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_estimators"      : [3],
            "criterion"         : ["gini", "entropy", "log_loss"],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
        	"min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
        	"max_features"      : [None, "sqrt", "log2"],
        	"bootstrap"         : [True, False],
        	"warm_start"        : [True, False],
        	"class_weight"      : [None],
            "random_state"      : [42]
            }
        
        self.machine_learning_grid_search(RandomForestClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "RandomForestClassifier")
        
    def run_extra_trees(self):
        """
        Run Extra Trees Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_estimators"      : [3],
            "criterion"         : ["gini", "entropy", "log_loss"],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
        	"min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
        	"max_features"      : [None, "sqrt", "log2"],
        	"bootstrap"         : [True, False],
        	"warm_start"        : [True, False],
        	"class_weight"      : [None],
            "random_state"      : [42]
            }
        
        self.machine_learning_grid_search(ExtraTreesClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "ExtraTreesClassifier")
        
    def run_grad_boosting(self):
        """
        Run Gradient Boosting Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "loss"              : ["log_loss", "exponential"],
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
        
        self.machine_learning_grid_search(GradientBoostingClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "GradientBoostingClassifier")
        
    def run_ada_boost(self):
        """
        Run Ada Boost Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "learning_rate" : [0.01, 0.1, 1.0, 10.0],
            "n_estimators"  : [150],
        	"algorithm"     : ["SAMME", "SAMME.R"],
            "random_state"  : [42]
            }
        
        self.machine_learning_grid_search(AdaBoostClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "AdaBoostClassifier")
        
    def run_MLP(self):
        """
        Run Multiple Layer Perceptor Classifier algorithm with grid search for hyperparameter tuning.
        """ 
        parameters = {
            "hidden_layer_sizes" : [(10,), (13,), (16,)],
            "activation"         : ["identity", "logistic", "tanh", "relu"],
            "solver"             : ["lbfgs", "sgd", "adam"],
            "learning_rate"      : ["constant", "invscaling", "adaptive"],
            "random_state"       : [42],
            "max_iter"           : [5000],
            "alpha"              : [0.0001,0.01,0.1],
            }
        
        self.machine_learning_grid_search(MLPClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "MLPClassifier")
        

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
                            scoring    = 'balanced_accuracy',
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
                                scoring=('accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'),
                                return_train_score=True)

        self.scores_train = np.mean(scores['train_balanced_accuracy'])
        self.scores_test  = np.mean(scores['test_balanced_accuracy'])

        print(model_name)
        print(f'Score on training set:   balanced_accuracy={self.scores_train * 100:.1f}%')
        print(f'Score on validation set: balanced_accuracy={self.scores_test * 100:.1f}%')
        print(' ')



#%% Main
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
    gkf = StratifiedGroupKFold(n_splits=5)
    
    # Initialize lists to store hyperparameters and mean scores for training and validation sets
    choose_hyperparam = []
    mean_scores_train = list()
    mean_scores_test  = list()
    
    # Iterate through cross-validation folds
    for train_ix, test_ix in gkf.split(data_processor.X_cal,data_processor.y_all,data_processor.groups):
        # Split data for the current fold
        X_train_val_a, X_test_a = X_features_a[train_ix, :],      X_features_a[test_ix, :]
        X_train_val_b, X_test_b = X_features_b[train_ix, :],      X_features_b[test_ix, :]
        X_train_val_c, X_test_c = X_features_c[train_ix, :],      X_features_c[test_ix, :]
        y_train_val,   y_test   = data_processor.y_all[train_ix], data_processor.y_all[test_ix]
        
        
        
        
        
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
        ml_algorithm = MLAlgorithm(X_ml   = features_extractor.X_ml, 
                                   y_ml   = features_extractor.y_ml, 
                                   ps     = features_extractor.ps, 
                                   n_jobs = 6)
    
        # Initialize the dictionary to map algorithm names to functions
        algorithm_mapping = {
            "knn":           ml_algorithm.run_knn,
            "gaussian_nb":   ml_algorithm.run_gaussian_nb,
            "bernoulli_nb":  ml_algorithm.run_bernoulli_nb,
            "decision_tree": ml_algorithm.run_decision_tree,
            "lda":           ml_algorithm.run_lda,
            "logreg":        ml_algorithm.run_logreg,
            "SVC":           ml_algorithm.run_SVC,
            "random_forest": ml_algorithm.run_random_forest,
            "extra_trees":   ml_algorithm.run_extra_trees,
            "grad_boosting": ml_algorithm.run_grad_boosting,
            "ada_boost":     ml_algorithm.run_ada_boost,
            "MLP":           ml_algorithm.run_MLP
        }
        
        # Specify the chosen algorithm (e.g., "knn", "gaussian_nb", "bernoulli_nb")
        chosen_algorithm = "decision_tree"  # Change this to select the desired algorithm
    
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
        
        # Delete variables to save memory
        del X_train_val_a, X_train_val_b, X_train_val_c, y_train_val, 
        del X_test_a, X_test_b, X_test_c, y_test, train_ix, test_ix
        
    # Print results
    print("")
    print("")
    print("Calibration for :  " + cal_case)
    print("Chosen Algorithm:  " + chosen_algorithm)
    print('mean_scores_train, Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_train)*100, np.std(mean_scores_train)*100))
    print('mean_scores_test,  Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_test)*100, np.std(mean_scores_test)*100))
    
    print("")
    for i in range(len(name_hyperparam)):
        print("Hyperparameter " + name_hyperparam[i] + " = " + str(most_common(choose_hyperparam[i])))

    
    
    
    