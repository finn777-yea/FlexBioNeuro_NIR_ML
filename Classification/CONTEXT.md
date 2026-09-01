# Classification

NIR spectroscopy classification workflows for biogas-related targets. Scripts are grouped into hyperparameter selection and held-out evaluation stages.

## Language

**Choose hyperparameter**:
The stage that searches for model hyperparameters using cross-validation on the training dataset.
_Avoid_: Hyperparam tuning, training stage

**Evaluate model**:
The stage that trains a fixed model configuration on the training dataset, then scores that same trained model on a separate held-out test dataset. The held-out test dataset is never used during training.
_Avoid_: Test run, validation stage

**Training dataset**:
The labeled NIR spectra used to fit the model.
_Avoid_: Calibration data, training-validation set, train set (in docs)

**Held-out test dataset**:
The reserved dataset used only after training, to score and plot Evaluate model results.
_Avoid_: Validation set, validation accuracy (for this dataset)

**Calibration case**:
The analyte or derived target selected for a run, such as `Ac_acid` or `VFA_TA`.
_Avoid_: Target type, acid case

**Absorbance calibration**:
The dark and reference scans used to convert raw intensity to absorbance.
_Avoid_: Calibration data, calibration file (without “absorbance”)

**DataProcessor**:
The component that loads NIR spectra, applies absorbance calibration, and prepares baseline-corrected feature matrices.
_Avoid_: Data loader, preprocessing pipeline

**Feature extractor**:
The component that scales spectra and reduces them into model-ready feature sets.
_Avoid_: Feature pipeline, Scaling_and_FeatureExtractor (in docs)

**Hyperparameter searcher**:
The component that searches over neural-network configuration settings for one cross-validation fold, using a backend such as genetic algorithm or particle swarm optimization.
_Avoid_: Hyperparam tuning, optimizer class, NeuralNetworkOptimizer (in docs)

**Model evaluator**:
The component that trains a fixed neural-network configuration on the training dataset and scores that same model on the held-out test dataset.
_Avoid_: NeuralNetworkOptimizer, optimizer class, MLAlgorithm (in docs)

**Spectral plotter**:
The component that visualizes class-separated NIR spectra for evaluation outputs.
_Avoid_: Plot helper, chart utility
