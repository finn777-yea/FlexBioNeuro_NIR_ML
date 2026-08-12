# Drop Keras-only NN search knobs for PyTorch migration

When migrating Classification Dense NN from Keras to PyTorch, several hyperparameters in the GA/PSO search space had no faithful PyTorch equivalent: activity regularization, Keras weight constraints (`MaxNorm`, `UnitNorm`, etc.), and optimizers such as `Ftrl`. We dropped those dimensions rather than silently no-op them, so search results remain interpretable and the PyTorch trainer only optimizes knobs it actually applies (`kernel_reg`, `bias_reg`, standard `torch.optim` classes).
