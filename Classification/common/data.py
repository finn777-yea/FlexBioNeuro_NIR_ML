import copy

import numpy as np
import pandas as pd
from scipy import signal


class DataProcessor:
    def __init__(self, data_file_path, calibration_file_path, meas_sec, test_data_file_path=None):
        """
        Initialize a DataProcessor object.

        Args:
            data_file_path (str): Path to the data file.
            calibration_file_path (str): Path to the calibration file.
            meas_sec (int): Measurement duration in seconds.
            test_data_file_path (str, optional): Path to the held-out test data file.
        """
        self.data_file_path = data_file_path
        self.test_data_file_path = test_data_file_path
        self.calibration_file_path = calibration_file_path
        self.meas_sec = meas_sec
        self.df_all = None
        self.X_all = None
        self.y_all = None
        self.groups = None
        self.X_cal = None
        self.X_features = None
        self.df_test = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        """Load preprocessed variables from a CSV data file."""
        self.df_all = pd.read_csv(self.data_file_path, encoding='ISO-8859-1')

    def load_test_data(self):
        """Load preprocessed variables from a CSV test data file."""
        self.df_test = pd.read_csv(self.test_data_file_path, encoding='ISO-8859-1')

    def set_XY_groups(self):
        """Set X and Y values and groups based on the loaded data."""
        self.X_all = self.df_all.iloc[:, :-1].values
        self.y_all = self.df_all.iloc[:, -1].values
        self.groups = list(
            np.repeat(np.arange(1, int(np.shape(self.X_all)[0] / self.meas_sec) + 1), self.meas_sec)
        )

    def set_XY_values(self):
        """Set X and Y values based on the loaded data."""
        self.X_all = self.df_all.iloc[:, :-1].values
        self.y_all = self.df_all.iloc[:, -1].values

        self.X_test = self.df_test.iloc[:, :-1].values
        self.y_test = self.df_test.iloc[:, -1].values

    def load_calibration(self, calibrate="yes"):
        """Load calibration data and calculate absorbance (X_cal)."""
        if calibrate == "yes":
            df_cal = pd.read_csv(self.calibration_file_path, encoding='ISO-8859-1')
            cal_dark = np.array(df_cal.iloc[0, :-1].values, dtype=np.float64)
            cal_ref = np.array(df_cal.iloc[1, :-1].values, dtype=np.float64)
            self.X_cal = -np.log10((self.X_all - cal_dark) / (cal_ref - cal_dark))
            if self.X_test is not None:
                self.X_test = -np.log10((self.X_test - cal_dark) / (cal_ref - cal_dark))
        else:
            self.X_cal = self.X_all

    def randomize(self):
        """Randomly shuffle the training data and test data when present."""
        permutation = np.random.permutation(int(self.y_all.shape[0] / self.meas_sec))
        permutation_group = np.repeat(permutation * 0, self.meas_sec)

        for i in range(len(permutation)):
            permutation_group[i * self.meas_sec:(i + 1) * self.meas_sec] = np.arange(
                permutation[i] * self.meas_sec, (permutation[i] + 1) * self.meas_sec
            )

        self.X_cal = self.X_cal[permutation_group, :]
        self.y_all = self.y_all[permutation_group]
        del permutation, permutation_group

        if self.X_test is None or self.y_test is None:
            return

        permutation = np.random.permutation(int(self.y_test.shape[0] / self.meas_sec))
        permutation_group = np.repeat(permutation * 0, self.meas_sec)

        for i in range(len(permutation)):
            permutation_group[i * self.meas_sec:(i + 1) * self.meas_sec] = np.arange(
                permutation[i] * self.meas_sec, (permutation[i] + 1) * self.meas_sec
            )

        self.X_test = self.X_test[permutation_group, :]
        self.y_test = self.y_test[permutation_group]
        del permutation, permutation_group

    def msc(self, input_data, reference=None):
        """Perform Multiplicative scatter correction."""
        for i in range(input_data.shape[0]):
            input_data[i, :] -= input_data[i, :].mean()

        if reference is None:
            ref = np.mean(input_data, axis=0)
        else:
            ref = reference

        data_msc = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            fit = np.polyfit(ref, input_data[i, :], 1, full=True)
            data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

        return (data_msc, ref)

    def snv(self, input_data):
        """Perform Standard Normal Variate (SNV) correction."""
        output_data = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])

        return output_data

    def split_blocks(self, X):
        """Split input data into blocks for different sensors."""
        len_S1_4 = len(range(1100, 1352, 2))
        len_S1_7 = len(range(1350, 1652, 2))
        len_S2_0 = len(range(1550, 1952, 2))
        len_S2_2 = len(range(1750, 2152, 2))

        end_S1_4 = len_S1_4
        end_S1_7 = len_S1_4 + len_S1_7
        end_S2_0 = len_S1_4 + len_S1_7 + len_S2_0
        end_S2_2 = len_S1_4 + len_S1_7 + len_S2_0 + len_S2_2

        X_S1_4 = X[:, 0:end_S1_4]
        X_S1_7 = X[:, end_S1_4:end_S1_7]
        X_S2_0 = X[:, end_S1_7:end_S2_0]
        X_S2_2 = X[:, end_S2_0:end_S2_2]

        return X_S1_4, X_S1_7, X_S2_0, X_S2_2

    def _baseline_correction_on_array(self, X, method, loop_sample_count):
        """Apply one baseline-correction method to a single spectra matrix.

        Args:
            X: Spectra to correct, shape (n_spectra, n_wavelengths).
            method: 1 = detrend, 2 = mean-center, 3 = SNV, 4 = MSC.
            loop_sample_count: Row count of *this* matrix ``X``, used by methods 3
                and 4 to walk groups of ``meas_sec`` spectra. Must be
                ``X.shape[0]``, not the training-dataset length when ``X`` is
                the held-out test dataset. Methods 1 and 2 ignore this argument.
        """
        X_S1_4, X_S1_7, X_S2_0, X_S2_2 = self.split_blocks(copy.deepcopy(X))

        if method == 1:
            X_S1_4 = signal.detrend(X_S1_4, axis=1)
            X_S1_7 = signal.detrend(X_S1_7, axis=1)
            X_S2_0 = signal.detrend(X_S2_0, axis=1)
            X_S2_2 = signal.detrend(X_S2_2, axis=1)
        elif method == 2:
            X_S1_4 = X_S1_4 - np.tile(np.mean(X_S1_4, axis=1), (X_S1_4.shape[1], 1)).T
            X_S1_7 = X_S1_7 - np.tile(np.mean(X_S1_7, axis=1), (X_S1_7.shape[1], 1)).T
            X_S2_0 = X_S2_0 - np.tile(np.mean(X_S2_0, axis=1), (X_S2_0.shape[1], 1)).T
            X_S2_2 = X_S2_2 - np.tile(np.mean(X_S2_2, axis=1), (X_S2_2.shape[1], 1)).T
        elif method == 3:
            for i in range(int(loop_sample_count / self.meas_sec)):
                X_S1_4[self.meas_sec * i:self.meas_sec * (i + 1), :] = self.snv(
                    X_S1_4[self.meas_sec * i:self.meas_sec * (i + 1), :]
                )
                X_S1_7[self.meas_sec * i:self.meas_sec * (i + 1), :] = self.snv(
                    X_S1_7[self.meas_sec * i:self.meas_sec * (i + 1), :]
                )
                X_S2_0[self.meas_sec * i:self.meas_sec * (i + 1), :] = self.snv(
                    X_S2_0[self.meas_sec * i:self.meas_sec * (i + 1), :]
                )
                X_S2_2[self.meas_sec * i:self.meas_sec * (i + 1), :] = self.snv(
                    X_S2_2[self.meas_sec * i:self.meas_sec * (i + 1), :]
                )
        elif method == 4:
            for i in range(int(loop_sample_count / self.meas_sec)):
                (X_S1_4[self.meas_sec * i:self.meas_sec * (i + 1), :], _) = self.msc(
                    X_S1_4[self.meas_sec * i:self.meas_sec * (i + 1), :]
                )
                (X_S1_7[self.meas_sec * i:self.meas_sec * (i + 1), :], _) = self.msc(
                    X_S1_7[self.meas_sec * i:self.meas_sec * (i + 1), :]
                )
                (X_S2_0[self.meas_sec * i:self.meas_sec * (i + 1), :], _) = self.msc(
                    X_S2_0[self.meas_sec * i:self.meas_sec * (i + 1), :]
                )
                (X_S2_2[self.meas_sec * i:self.meas_sec * (i + 1), :], _) = self.msc(
                    X_S2_2[self.meas_sec * i:self.meas_sec * (i + 1), :]
                )

        return np.hstack((X_S1_4, X_S1_7, X_S2_0, X_S2_2))

    def baseline_correction(self, method):
        """
        Apply baseline correction to training data and test data when present.

        Returns:
            tuple: (train_features, test_features). test_features is None when no test set exists.
        """
        X_train = self._baseline_correction_on_array(self.X_cal, method, self.X_cal.shape[0])

        if self.X_test is None:
            return X_train, None

        X_test = self._baseline_correction_on_array(self.X_test, method, self.X_test.shape[0])
        return X_train, X_test
