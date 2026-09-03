# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:45:51 2023

@author: ge23hum
"""


from pathlib import Path
import argparse
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Classification/, for `common`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `paths`

from paths import calibration_path, data_path

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

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 128

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

CAL_CASE_FILES = {
    "VFA_TA": {
        "data_file": "classification_NIR_Data_raw_VFA_TA.csv",
        "test_data_file": "Test_classification_NIR_Data_raw_VFA_TA.csv",
        "calibrate": "no",
    },
    "Ac_acid": {
        "data_file": "classification_NIR_Data_raw_Ac_acid.csv",
        "test_data_file": "Test_classification_NIR_Data_raw_Ac_acid.csv",
        "calibrate": "yes",
    },
}


class Tee:
    """Write to multiple streams and flush immediately (terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")


def start_tee_log(cal_case):
    """Tee stdout/stderr to a timestamped log under logs/ and return the path and file."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"torch_NN_GA_eval_{cal_case}_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_path, log_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed Torch NN GA configuration on a held-out test dataset."
    )
    parser.add_argument(
        "--cal-case",
        dest="cal_case",
        choices=sorted(CAL_CASE_FILES),
        default="VFA_TA",
        help="Calibration case (analyte) to evaluate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size to train the model.",
    )
    parser.add_argument(
        "--show-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show confusion matrix, learning curve, and ROC plots.",
    )
    return parser.parse_args()


class NnEvaluator:
    """Train a fixed network on the training dataset and score the held-out test dataset."""

    def __init__(self, X_train, y_train, X_test, y_test, cal_case, epochs, batch_size, show_plot):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.show_plot = show_plot
        self.cal_case = cal_case
        self.model = None
        self.history = None
        self.hyperparams = EVAL_CONFIGS[cal_case]
        self.model_builder = BaselineNnBuilder()

    def create_baseline(self):
        """Create a baseline dense network for the selected calibration case."""
        return self.model_builder.build(np.shape(self.X_train)[1], **self.hyperparams)

    def evaluate_nn(self):
        """Train once on the training dataset, then score the held-out test dataset."""
        self.model = self.create_baseline()
        self.model, self.history = train_model(
            self.model,
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            kernel_reg=self.hyperparams["kernel_reg"],
            bias_reg=self.hyperparams["bias_reg"],
            optim=self.hyperparams["optim"],
            return_history=True,
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
        index = ["training dataset", "held-out test dataset"]
        columns = ["balanced_accuracy", "precision", "recall", "f1-score"]
        df_scores = pd.DataFrame(data, index, columns)
        print(model_name)
        print(df_scores)
        print(" ")

    def plot_learning_curve(self, model_name):
        if not self.show_plot:
            return

        accuracy = self.history["binary_accuracy"]
        epochs = range(1, len(accuracy) + 1)

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
        ax.set_ylabel("Accuracy", fontsize=fontsize_ylabel)

        ax.grid(True, linewidth=grid_line_width, ls="--")
        ax.plot(epochs, accuracy, color="magenta", label="Training accuracy")
        ax.legend(loc="lower right", fontsize=fontsize_legend)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, model_name):
        if not self.show_plot:
            return

        y_pred_test = predict_classes(self.model, self.X_test)
        cf_matrix = confusion_matrix(self.y_test, y_pred_test)

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

        if not self.show_plot:
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
    args = parse_args()
    cal_case = args.cal_case
    case_files = CAL_CASE_FILES[cal_case]

    log_path, log_file = start_tee_log(cal_case)
    if torch.cuda.is_available():
        device_line = "cuda (%s)" % torch.cuda.get_device_name(0)
    else:
        device_line = "CUDA not available"

    print("=" * 72)
    print("Evaluate model: Torch NN GA")
    print("Log file:           " + str(log_path))
    print("Calibration case:   " + cal_case)
    print("Device:             " + device_line)
    print("Epochs:             " + str(args.epochs))
    print("Batch size:         " + str(args.batch_size))
    print("Show plot:          " + ("yes" if args.show_plot else "no"))
    print("=" * 72)

    try:
        data_processor = DataProcessor(
            data_file_path=data_path(case_files["data_file"]),
            test_data_file_path=data_path(case_files["test_data_file"]),
            calibration_file_path=calibration_path(),
            meas_sec=8,
        )

        data_processor.load_data()
        data_processor.load_test_data()
        data_processor.set_XY_values()
        data_processor.load_calibration(calibrate=case_files["calibrate"])
        data_processor.randomize()
        X_train_a, X_test_a = data_processor.baseline_correction(method=2)
        X_train_b, X_test_b = data_processor.baseline_correction(method=3)
        X_train_c, X_test_c = data_processor.baseline_correction(method=4)

        features_extractor = Scaling_and_FeatureExtractor(
            number_of_features=5,
            X_train_val_a=X_train_a,
            X_train_val_b=X_train_b,
            X_train_val_c=X_train_c,
            y_train_val=data_processor.y_all,
            X_test_a=X_test_a,
            X_test_b=X_test_b,
            X_test_c=X_test_c,
            y_test=data_processor.y_test,
            df_all=data_processor.df_all,
        )

        features_extractor.Scaling_data()
        features_extractor.extract_features()

        evaluator = NnEvaluator(
            X_train=features_extractor.X_out,
            y_train=features_extractor.y_train_val,
            X_test=features_extractor.X_out_test,
            y_test=features_extractor.y_test,
            cal_case=cal_case,
            epochs=args.epochs,
            batch_size=args.batch_size,
            show_plot=args.show_plot,
        )

        evaluator.evaluate_nn()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
