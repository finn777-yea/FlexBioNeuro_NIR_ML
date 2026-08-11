import copy

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import MinMaxScaler


class Scaling_and_FeatureExtractor:
    def __init__(
        self,
        number_of_features,
        X_train_val_a,
        X_train_val_b,
        X_train_val_c,
        y_train_val,
        X_test_a,
        X_test_b,
        X_test_c,
        y_test,
        df_all,
        meas_sec=None,
    ):
        self.number_of_features = number_of_features
        self.X_train_val_a = X_train_val_a
        self.X_train_val_b = X_train_val_b
        self.X_train_val_c = X_train_val_c
        self.y_train_val = y_train_val
        self.X_test_a = X_test_a
        self.X_test_b = X_test_b
        self.X_test_c = X_test_c
        self.y_test = y_test
        self.df_all = df_all
        self.meas_sec = meas_sec
        self.X_out = None
        self.X_out_test = None
        self.X_ml = None
        self.y_ml = None
        self.ps = None

    def Scaling_data(self):
        """Scale input data using MinMaxScaler."""
        scaler_train_val_a = MinMaxScaler(feature_range=(0, 1))
        self.X_train_val_a = scaler_train_val_a.fit_transform(self.X_train_val_a)
        self.X_test_a = scaler_train_val_a.transform(self.X_test_a)

        scaler_train_val_b = MinMaxScaler(feature_range=(0, 1))
        self.X_train_val_b = scaler_train_val_b.fit_transform(self.X_train_val_b)
        self.X_test_b = scaler_train_val_b.transform(self.X_test_b)

        scaler_train_val_c = MinMaxScaler(feature_range=(0, 1))
        self.X_train_val_c = scaler_train_val_c.fit_transform(self.X_train_val_c)
        self.X_test_c = scaler_train_val_c.transform(self.X_test_c)

    def extra_trees_feature_reduction(self, X_train_val, y_train_val, df_all, X_test):
        """Perform feature reduction using Extra Trees Classifier."""
        model = ExtraTreesClassifier()
        model.fit(X_train_val, y_train_val)
        feat_importances = pd.Series(model.feature_importances_, index=df_all.columns[:-1])

        dfcolumns_out = []
        best_feature_importance = feat_importances.nlargest(self.number_of_features)
        ind_best_feature = np.argwhere(df_all.columns[:-1] == best_feature_importance.index[0])[0][0]
        X_out_FR = copy.deepcopy(X_train_val[:, ind_best_feature])
        X_out_test_FR = copy.deepcopy(X_test[:, ind_best_feature])
        dfcolumns_out.append(df_all.columns[ind_best_feature])
        for i in range(self.number_of_features - 1):
            ind_best_feature = np.argwhere(df_all.columns[:-1] == best_feature_importance.index[i + 1])[0][0]
            X_out_FR = np.column_stack((X_out_FR, X_train_val[:, ind_best_feature]))
            X_out_test_FR = np.column_stack((X_out_test_FR, X_test[:, ind_best_feature]))
            dfcolumns_out.append(df_all.columns[ind_best_feature])

        return X_out_FR, X_out_test_FR, dfcolumns_out

    def univariate_selection(self, X_train_val, y_train_val, df_all, X_test):
        """Perform feature selection using the ANOVA F-test."""
        bestfeatures = SelectKBest(score_func=f_classif, k=self.number_of_features)
        fit = bestfeatures.fit(X_train_val, y_train_val)
        dfcolumns = pd.DataFrame(df_all.columns[:-1])
        dfscores = pd.DataFrame(fit.scores_)

        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']

        dfcolumns_out = []
        best_univariate_selection = featureScores.nlargest(self.number_of_features, 'Score')
        ind_best_feature = best_univariate_selection.index[0]
        X_out_US = copy.deepcopy(X_train_val[:, ind_best_feature])
        X_out_test_US = copy.deepcopy(X_test[:, ind_best_feature])
        dfcolumns_out.append(df_all.columns[ind_best_feature])
        for i in range(1, self.number_of_features):
            ind_best_feature = best_univariate_selection.index[i]
            X_out_US = np.column_stack((X_out_US, X_train_val[:, ind_best_feature]))
            X_out_test_US = np.column_stack((X_out_test_US, X_test[:, ind_best_feature]))
            dfcolumns_out.append(df_all.columns[ind_best_feature])

        return X_out_US, X_out_test_US, dfcolumns_out

    def pls(self, X_train_val, y_train_val, X_test):
        """Perform feature reduction using Partial Least Squares (PLS)."""
        plsda = PLSRegression(n_components=self.number_of_features)
        (X_out_PLS, _) = plsda.fit_transform(X_train_val, y_train_val)
        X_out_test_PLS = plsda.transform(X_test)

        return X_out_PLS, X_out_test_PLS

    def pca(self, X_train_val, X_test):
        """Perform feature reduction using Principal Component Analysis (PCA)."""
        sklearn_pca = PCA(n_components=self.number_of_features)
        X_out_PCA = sklearn_pca.fit_transform(X_train_val)
        X_out_test_PCA = sklearn_pca.transform(X_test)

        return X_out_PCA, X_out_test_PCA

    def truncated_svd(self, X_train_val, X_test):
        """Perform feature reduction using Truncated Singular Value Decomposition (TruncatedSVD)."""
        sklearn_svd = TruncatedSVD(n_components=self.number_of_features)
        X_out_SVD = sklearn_svd.fit_transform(X_train_val)
        X_out_test_SVD = sklearn_svd.transform(X_test)

        return X_out_SVD, X_out_test_SVD

    def extract_features(self):
        """Extract and reduce features using various methods."""
        X_out_FR, X_out_test_FR, _ = self.extra_trees_feature_reduction(
            self.X_train_val_a, self.y_train_val, self.df_all, self.X_test_a
        )
        X_out_US, X_out_test_US, _ = self.univariate_selection(
            self.X_train_val_a, self.y_train_val, self.df_all, self.X_test_a
        )
        X_out_PLS, X_out_test_PLS = self.pls(self.X_train_val_a, self.y_train_val, self.X_test_a)
        X_out_PCA, X_out_test_PCA = self.pca(self.X_train_val_a, self.X_test_a)
        X_out_SVD, X_out_test_SVD = self.truncated_svd(self.X_train_val_a, self.X_test_a)

        X_out_a = np.hstack((X_out_FR, X_out_US, X_out_PLS, X_out_PCA, X_out_SVD))
        X_out_test_a = np.hstack((X_out_test_FR, X_out_test_US, X_out_test_PLS, X_out_test_PCA, X_out_test_SVD))

        del X_out_FR, X_out_test_FR, X_out_US, X_out_test_US, X_out_PLS, X_out_test_PLS
        del X_out_PCA, X_out_test_PCA, X_out_SVD, X_out_test_SVD

        X_out_FR, X_out_test_FR, _ = self.extra_trees_feature_reduction(
            self.X_train_val_b, self.y_train_val, self.df_all, self.X_test_b
        )
        X_out_US, X_out_test_US, _ = self.univariate_selection(
            self.X_train_val_b, self.y_train_val, self.df_all, self.X_test_b
        )
        X_out_PLS, X_out_test_PLS = self.pls(self.X_train_val_b, self.y_train_val, self.X_test_b)
        X_out_PCA, X_out_test_PCA = self.pca(self.X_train_val_b, self.X_test_b)
        X_out_SVD, X_out_test_SVD = self.truncated_svd(self.X_train_val_b, self.X_test_b)

        X_out_b = np.hstack((X_out_FR, X_out_US, X_out_PLS, X_out_PCA, X_out_SVD))
        X_out_test_b = np.hstack((X_out_test_FR, X_out_test_US, X_out_test_PLS, X_out_test_PCA, X_out_test_SVD))

        del X_out_FR, X_out_test_FR, X_out_US, X_out_test_US, X_out_PLS, X_out_test_PLS
        del X_out_PCA, X_out_test_PCA, X_out_SVD, X_out_test_SVD

        X_out_FR, X_out_test_FR, _ = self.extra_trees_feature_reduction(
            self.X_train_val_c, self.y_train_val, self.df_all, self.X_test_c
        )
        X_out_US, X_out_test_US, _ = self.univariate_selection(
            self.X_train_val_c, self.y_train_val, self.df_all, self.X_test_c
        )
        X_out_PLS, X_out_test_PLS = self.pls(self.X_train_val_c, self.y_train_val, self.X_test_c)
        X_out_PCA, X_out_test_PCA = self.pca(self.X_train_val_c, self.X_test_c)
        X_out_SVD, X_out_test_SVD = self.truncated_svd(self.X_train_val_c, self.X_test_c)

        X_out_c = np.hstack((X_out_FR, X_out_US, X_out_PLS, X_out_PCA, X_out_SVD))
        X_out_test_c = np.hstack((X_out_test_FR, X_out_test_US, X_out_test_PLS, X_out_test_PCA, X_out_test_SVD))

        del X_out_FR, X_out_test_FR, X_out_US, X_out_test_US, X_out_PLS, X_out_test_PLS
        del X_out_PCA, X_out_test_PCA, X_out_SVD, X_out_test_SVD

        self.X_out = np.hstack((X_out_a, X_out_b, X_out_c))
        self.X_out_test = np.hstack((X_out_test_a, X_out_test_b, X_out_test_c))

    def Predefined_cv(self):
        """Perform predefined cross-validation."""
        train_indices = np.full((np.size(self.y_train_val),), -1, dtype=int)
        val_indices = np.full((np.size(self.y_test),), 0, dtype=int)
        test_fold = np.append(train_indices, val_indices)

        self.ps = PredefinedSplit(test_fold)
        self.X_ml = np.concatenate((self.X_out, self.X_out_test))
        self.y_ml = np.concatenate((self.y_train_val, self.y_test))

    def reshape_data(self):
        """Reshape input data arrays for recurrent neural networks."""
        if self.meas_sec is None:
            raise ValueError("meas_sec is required for reshape_data")

        self.X_out = self.X_out.reshape(
            int(self.X_out.shape[0] / self.meas_sec), self.meas_sec, self.X_out.shape[1]
        )
        self.y_train_val = self.y_train_val.reshape(len(self.y_train_val), 1)[:: self.meas_sec]

        self.X_out_test = self.X_out_test.reshape(
            int(self.X_out_test.shape[0] / self.meas_sec), self.meas_sec, self.X_out_test.shape[1]
        )
        self.y_test = self.y_test.reshape(len(self.y_test), 1)[:: self.meas_sec]
