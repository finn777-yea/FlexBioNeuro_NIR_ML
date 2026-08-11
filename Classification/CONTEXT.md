# Classification

NIR spectroscopy classification workflows for biogas-related targets. Scripts are grouped into hyperparameter selection and held-out evaluation stages.

## Language

**Choose hyperparameter**:
The stage that searches for model hyperparameters using cross-validation on calibration data.
_Avoid_: Hyperparam tuning, training stage

**Evaluate model**:
The stage that scores a fixed model configuration on a separate held-out test dataset.
_Avoid_: Test run, validation stage

**Calibration case**:
The analyte or derived target selected for a run, such as `Ac_acid` or `VFA_TA`.
_Avoid_: Target type, acid case

**DataProcessor**:
The component that loads NIR spectra, applies calibration, and prepares baseline-corrected feature matrices.
_Avoid_: Data loader, preprocessing pipeline

**Feature extractor**:
The component that scales spectra and reduces them into model-ready feature sets.
_Avoid_: Feature pipeline, Scaling_and_FeatureExtractor (in docs)

**Hyperparameter searcher**:
The component that searches over neural-network configuration settings for one cross-validation fold, using a backend such as genetic algorithm or particle swarm optimization.
_Avoid_: Hyperparam tuning, optimizer class, NeuralNetworkOptimizer (in docs)

**Spectral plotter**:
The component that visualizes class-separated NIR spectra for evaluation outputs.
_Avoid_: Plot helper, chart utility
