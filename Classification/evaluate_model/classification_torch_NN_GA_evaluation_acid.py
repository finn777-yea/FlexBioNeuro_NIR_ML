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
from common.nn_model import BaselineNnBuilder
from common.nn_train import predict_classes, predict_proba, train_model

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.warn = warn
warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

EVAL_CONFIGS = {
    "VFA_TA": {
        "neuron_layer_1": 12,
        "neuron_layer_2": 9,
        "activation_1": "relu",
        "activation_2": "relu",
        "kernel_init": 1,
        "bias_init": 3,
        "kernel_reg": "L1L2",
        "bias_reg": "L2",
        "optim": "Nadam",
    },
    "Ac_acid": {
        "neuron_layer_1": 12,
        "neuron_layer_2": 8,
        "activation_1": "relu",
        "activation_2": "sigmoid",
        "kernel_init": 1,
        "bias_init": 4,
        "kernel_reg": "L1",
        "bias_reg": "L1L2",
        "optim": "Adagrad",
    },
}


class NeuralNetworkOptimizer:
    def __init__(self, X_ml, y_ml, ps, meas_sec, cal_case, epochs=200, show_plot="no"):
        self.X_ml = X_ml
        self.y_ml = y_ml
        self.ps = ps
        self.meas_sec = meas_sec
        self.epochs = epochs
        self.show_plot = show_plot
        self.cal_case = cal_case
        self.X_train = self.X_ml[self.ps.test_fold == -1]
        self.y_train = self.y_ml[self.ps.test_fold == -1]
        self.X_test = self.X_ml[self.ps.test_fold == 0]
        self.y_test = self.y_ml[self.ps.test_fold == 0]
        self.model = None
        self.hyperparams = EVAL_CONFIGS[cal_case]
        self.model_builder = BaselineNnBuilder()

    def create_baseline(self):
        """Create a baseline dense network for the selected calibration case."""
        return self.model_builder.build(np.shape(self.X_train)[1], **self.hyperparams)

    def evaluate_nn(self):
        """Train and evaluate the deep neural network model."""
        self.model = self.create_baseline()
        self.model = train_model(
            self.model,
            self.X_train,
            self.y_train,
            X_val=self.X_test,
            y_val=self.y_test,
            epochs=self.epochs,
            batch_size=32,
            kernel_reg=self.hyperparams["kernel_reg"],
            bias_reg=self.hyperparams["bias_reg"],
            optim=self.hyperparams["optim"],
        )

        self.machine_learning_best_scores("Deep Neural Network")
        self.plot_confusion_matrix("Deep Neural Network")
        self.plot_learning_curve("Deep Neural Network")
        self.plot_ROC_curves_and_AUC("Deep Neural Network")

    def machine_learning_best_scores(self, model_name):
        preds_train = predict_classes(self.model, self.X_train)
        preds_test = predict_classes(self.model, self.X_test)

        train1 = balanced_accuracy_score(self.y_train, preds_train)
        test1 = balanced_accuracy_score(self.y_test, preds_test)

        train2 = precision_score(self.y_train, preds_train)
        test2 = precision_score(self.y_test, preds_test)

        train3 = recall_score(self.y_train, preds_train)
        test3 = recall_score(self.y_test, preds_test)

        train4 = f1_score(self.y_train, preds_train)
        test4 = f1_score(self.y_test, preds_test)

        data = [
            [f"{train1*100:.2f}%", f"{train2*100:.2f}%", f"{train3*100:.2f}%", f"{train4*100:.2f}%"],
            [f"{test1*100:.2f}%", f"{test2*100:.2f}%", f"{test3*100:.2f}%", f"{test4*100:.2f}%"],
        ]
        index = ["training-validation set", "test set"]
        columns = ["balanced_accuracy", "precision", "recall", "f1-score"]
        df_scores = pd.DataFrame(data, index, columns)
        print(model_name)
        print(df_scores)
        print(" ")

    def plot_learning_curve(self, model_name):
        if self.show_plot != "yes":
            return

        _, history = train_model(
            self.create_baseline(),
            self.X_train,
            self.y_train,
            X_val=self.X_test,
            y_val=self.y_test,
            epochs=self.epochs,
            batch_size=32,
            kernel_reg=self.hyperparams["kernel_reg"],
            bias_reg=self.hyperparams["bias_reg"],
            optim=self.hyperparams["optim"],
            return_history=True,
        )

        loss_values = history["loss"]
        accuracy = history["binary_accuracy"]
        val_accuracy = history["val_binary_accuracy"]
        epochs = range(1, len(loss_values) + 1)

        fontsize_title = 10
        fontsize_xlabel = 8
        fontsize_ylabel = 8
        fontsize_xtick = 6
        fontsize_ytick = 6
        fontsize_legend = 6
        grid_line_width = 0.5

        plt.rc("font", family="serif")
        plt.rc("xtick", labelsize=fontsize_xtick)
        plt.rc("ytick", labelsize=fontsize_ytick)
        plt.rcParams["figure.dpi"] = 450
        plt.rcParams["figure.autolayout"] = True

        fig = plt.figure(figsize=(4.00, 3.00))
        ax = fig.subplots()

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_tick_params(which="major", size=5, width=1.0, direction="in", top=False)
        ax.yaxis.set_tick_params(which="major", size=5, width=1.0, direction="in", right=False)

        ax.set_title(model_name, fontsize=fontsize_title)
        ax.set_xlabel("Epochs", fontsize=fontsize_xlabel)
        ax.set_ylabel("Score", fontsize=fontsize_ylabel)

        ax.grid(True, linewidth=grid_line_width, ls="--")
        ax.plot(epochs, accuracy, color="magenta", label="Training accuracy")
        ax.plot(epochs, val_accuracy, color="green", label="Validation accuracy")
        ax.legend(loc="lower right", fontsize=fontsize_legend)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, model_name):
        y_pred_test = predict_classes(self.model, self.X_test)
        cf_matrix = confusion_matrix(self.y_test, y_pred_test)

        if self.show_plot != "yes":
            return

        plt.figure(figsize=(4.00, 3.00))
        cmn = cf_matrix.astype("int")
        ax = sns.heatmap(cmn, annot=True, vmin=0, vmax=cf_matrix.max(), cmap="YlGnBu")

        ax.set_title(model_name)
        ax.set_xlabel("\nPredicted Values")
        ax.set_ylabel("Actual Values ")
        ax.xaxis.set_ticklabels(["Class 0", "Class 1"])
        ax.yaxis.set_ticklabels(["Class 0", "Class 1"])

        plt.tight_layout()
        plt.show()

    def plot_ROC_curves_and_AUC(self, model_name):
        ns_probs = [0 for _ in range(len(self.y_test))]
        lr_probs = predict_proba(self.model, self.X_test)

        ns_auc = roc_auc_score(self.y_test, ns_probs)
        lr_auc = roc_auc_score(self.y_test, lr_probs)

        print("Baseline: ROC AUC=%.3f" % (ns_auc))
        print("ML algorithm: ROC AUC=%.3f" % (lr_auc))

        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, lr_probs)

        if self.show_plot != "yes":
            return

        fontsize_title = 10
        fontsize_xlabel = 8
        fontsize_ylabel = 8
        fontsize_xtick = 6
        fontsize_ytick = 6
        fontsize_legend = 6
        grid_line_width = 0.5

        plt.rc("font", family="serif")
        plt.rc("xtick", labelsize=fontsize_xtick)
        plt.rc("ytick", labelsize=fontsize_ytick)
        plt.rcParams["figure.dpi"] = 450
        plt.rcParams["figure.autolayout"] = True

        fig = plt.figure(figsize=(4.00, 3.00))
        ax = fig.subplots()

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_tick_params(which="major", size=5, width=1.0, direction="in", top=False)
        ax.yaxis.set_tick_params(which="major", size=5, width=1.0, direction="in", right=False)

        ax.set_title(model_name, fontsize=fontsize_title)
        ax.set_xlabel("False Positive Rate", fontsize=fontsize_xlabel)
        ax.set_ylabel("True Positive Rate", fontsize=fontsize_ylabel)

        ax.grid(True, linewidth=grid_line_width, ls="--")
        ax.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill", color="magenta")
        ax.plot(lr_fpr, lr_tpr, marker=".", markersize=3, label="ML-model", color="green")

        text_AUC = "AUC = %.3f" % (lr_auc)
        ax.text(0.7, 0.5, text_AUC, color="grey")
        ax.legend(loc="lower right", fontsize=fontsize_legend)

        plt.tight_layout()
        plt.show()


#%% Main
if __name__ == "__main__":
    cal_case = "VFA_TA"
    if cal_case == "VFA_TA":
        data_file_path = "Data/classification_NIR_Data_raw_VFA_TA.csv"
        test_data_file_path = "Data/Test_classification_NIR_Data_raw_VFA_TA.csv"
        calibrate = "no"
    elif cal_case == "Ac_acid":
        data_file_path = "Data/classification_NIR_Data_raw_Ac_acid.csv"
        test_data_file_path = "Data/Test_classification_NIR_Data_raw_Ac_acid.csv"
        calibrate = "yes"

    data_processor = DataProcessor(
        data_file_path=data_file_path,
        test_data_file_path=test_data_file_path,
        calibration_file_path="Data/calibration_NIR_Data.csv",
        meas_sec=8,
    )

    data_processor.load_data()
    data_processor.load_test_data()
    data_processor.set_XY_values()
    data_processor.load_calibration(calibrate=calibrate)
    data_processor.randomize()
    X_train_val_a, X_test_a = data_processor.baseline_correction(method=2)
    X_train_val_b, X_test_b = data_processor.baseline_correction(method=3)
    X_train_val_c, X_test_c = data_processor.baseline_correction(method=4)

    features_extractor = Scaling_and_FeatureExtractor(
        number_of_features=5,
        X_train_val_a=X_train_val_a,
        X_train_val_b=X_train_val_b,
        X_train_val_c=X_train_val_c,
        y_train_val=data_processor.y_all,
        X_test_a=X_test_a,
        X_test_b=X_test_b,
        X_test_c=X_test_c,
        y_test=data_processor.y_test,
        df_all=data_processor.df_all,
    )

    features_extractor.Scaling_data()
    features_extractor.extract_features()
    features_extractor.Predefined_cv()

    nn_optimizer = NeuralNetworkOptimizer(
        X_ml=features_extractor.X_ml,
        y_ml=features_extractor.y_ml,
        ps=features_extractor.ps,
        meas_sec=data_processor.meas_sec,
        epochs=500,
        show_plot="yes",
        cal_case=cal_case,
    )

    nn_optimizer.evaluate_nn()
