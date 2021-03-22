import tensorflow as tf
from tensorflow import keras


class Model(keras.Model):
    def __init__(
        self, number_of_states: int, hidden_units: list, number_of_actions: int
    ):
        super(Model, self).__init__()
        self.input_layer = keras.layers.InputLayer(input_shape=(number_of_states))
        self.hidden_layers = []

        weights_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.1)

        for hidden_unit in hidden_units:
            self.hidden_layers.append(
                keras.layers.Dense(
                    hidden_unit,
                    activation="relu",
                    kernel_initializer=weights_initializer,
                )
            )

        self.output_layer = keras.layers.Dense(
            number_of_actions,
            activation="linear",
            kernel_initializer=weights_initializer,
        )

    @tf.function
    def call(self, inputs):
        outputFromLayer = self.input_layer(inputs)
        for layer in self.hidden_layers:
            outputFromLayer = layer(outputFromLayer)
        outputFromModel = self.output_layer(outputFromLayer)
        return outputFromModel
