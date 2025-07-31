from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras


class AbstractMethod(ABC):
    @abstractmethod
    def _build_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, input, target):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass


class CBOW(AbstractMethod):
    def __init__(self, vocab_size, window_size=3, embedding_size=10, epoch=100) -> None:
        self._model = self._build_model(vocab_size, embedding_size, window_size)
        self.epoch = epoch

    def _build_model(self, input_size: int, embedding_size: int, window_size: int):
        inputs = keras.Input(shape=(2 * window_size,))
        x = keras.layers.Embedding(
            input_dim=input_size,
            output_dim=embedding_size,
            input_length=2 * window_size,
        )(inputs)
        x = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
        outputs = keras.layers.Dense(units=input_size, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, input, target):
        history = self._model.fit(input, target, epochs=self.epoch, verbose=0)
        return history.history["loss"][0]

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)


class SkipGram(AbstractMethod):
    def __init__(self, vocab_size, embedding_size=10, epoch=100) -> None:
        self._model = self._build_model(vocab_size, embedding_size)
        self.epoch = epoch

    def _build_model(self, vocab_size: int, embedding_size: int):
        inputs = keras.Input(shape=(1,))
        x = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            input_length=1,
        )(inputs)
        x = keras.layers.Reshape((embedding_size,))(x)
        outputs = keras.layers.Dense(units=vocab_size, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, input, target):
        history = self._model.fit(input, target, epochs=self.epoch, verbose=0)
        return history.history["loss"][0]

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)
