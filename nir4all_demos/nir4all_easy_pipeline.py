import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, train_test_split
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `paths`

from paths import data_path

from nirs4all.visualization.predictions import PredictionAnalyzer
import matplotlib.pyplot as plt

DATA_PATH = data_path("classification_NIR_Data_raw_Ac_acid.csv")

data = pd.read_csv(DATA_PATH)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_np = np.array(X).astype(float)
y_np = np.array(y).astype(float)

print(f"{DATA_PATH} loaded successfully")
print(f"X shape: {X_np.shape}")
print(f"y shape: {y_np.shape}")

# Hold out a test set. plot_confusion_matrix defaults to display_partition='test';
# passing only (X, y) marks every sample as train, so the figure is empty.
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.25, stratify=y_np, random_state=42
)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

pipeline = [
    MinMaxScaler(),                              # Scale features to [0, 1]
    ShuffleSplit(n_splits=3, test_size=0.25),    # 3-fold cross-validation
    {"model": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)}
]

results = nirs4all.run(
    pipeline=pipeline,
    dataset={
        "name": "acetic_acid",
        "train_x": X_train,
        "train_y": y_train,
        "test_x": X_test,
        "test_y": y_test,
    },
    name="MyFirstPipeline",
    verbose=1
)

print(f"\n📊 Results:")
print(f"   Best score: {results.best_score:.4f}")
print(f"   Best accuracy: {results.best_accuracy:.4f}")
print(f"   Total predictions: {results.num_predictions}")

analyzer = PredictionAnalyzer(results.predictions)
fig1 = analyzer.plot_confusion_matrix(
    k=3,
    rank_metric="balanced_accuracy",
    rank_partition="val",
    display_partition="test",
)
plt.show()