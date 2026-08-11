import numpy as np
from keras import initializers
from keras.layers import Dense
from keras.models import Sequential


class BaselineNnBuilder:
    """Build and compile a 3-layer Dense binary classifier from hyperparameters."""

    def __init__(self, seed=42):
        self.keras_init = [
            initializers.RandomNormal(seed=seed),
            initializers.RandomUniform(seed=seed),
            initializers.TruncatedNormal(seed=seed),
            initializers.VarianceScaling(seed=seed),
            initializers.GlorotNormal(seed=seed),
            initializers.GlorotUniform(seed=seed),
            initializers.HeNormal(seed=seed),
            initializers.HeUniform(seed=seed),
            initializers.LecunNormal(seed=seed),
            initializers.LecunUniform(seed=seed),
        ]

    def build(
        self,
        input_dim,
        neuron_layer_1=15,
        neuron_layer_2=8,
        activation_1="elu",
        activation_2="elu",
        activation_3="sigmoid",
        kernel_init=7,
        bias_init=5,
        kernel_reg=None,
        bias_reg=None,
        activity_reg=None,
        kernel_const=None,
        bias_const=None,
        optim="Adam",
    ):
        """Create and compile a Keras Sequential model with the given hyperparameters."""
        model = Sequential()
        model.add(
            Dense(
                units=neuron_layer_1,
                input_dim=input_dim,
                activation=activation_1,
                kernel_initializer=self.keras_init[kernel_init],
                bias_initializer=self.keras_init[bias_init],
                kernel_regularizer=kernel_reg,
                bias_regularizer=bias_reg,
                activity_regularizer=activity_reg,
                kernel_constraint=kernel_const,
                bias_constraint=bias_const,
            )
        )

        model.add(
            Dense(
                units=neuron_layer_2,
                activation=activation_2,
                kernel_initializer=self.keras_init[kernel_init],
                bias_initializer=self.keras_init[bias_init],
                kernel_regularizer=kernel_reg,
                bias_regularizer=bias_reg,
                activity_regularizer=activity_reg,
                kernel_constraint=kernel_const,
                bias_constraint=bias_const,
            )
        )

        model.add(
            Dense(
                units=1,
                activation=activation_3,
                kernel_initializer=self.keras_init[kernel_init],
                bias_initializer=self.keras_init[bias_init],
                kernel_regularizer=kernel_reg,
                bias_regularizer=bias_reg,
                activity_regularizer=activity_reg,
                kernel_constraint=kernel_const,
                bias_constraint=bias_const,
            )
        )

        model.compile(loss="binary_crossentropy", optimizer=optim, metrics=["binary_accuracy"])
        return model
