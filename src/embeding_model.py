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

    def _build_model(self, vocab_size: int, embedding_size: int, window_size: int):
        inputs = keras.Input(shape=(vocab_size,))
        x = keras.layers.Dense(embedding_size, activation="linear")(inputs)
        outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, input, target):
        input = np.array(input)  # shape: (3955, vocab_size)
        target = np.array(target)  # shape: (3955, vocab_size)
        print(input.shape, target.shape)
        print(
            f"token_list size {len(input)}  input shape {input.shape} input shape {input[0].shape}sum {np.sum(input[0])}"
        )
        history = self._model.fit(input, target, verbose=1, epochs=self.epoch)
        return history.history["loss"]

    def predict(self, input):
        input = np.array(input)
        return self._model.predict(input)

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)


class CBOW2(AbstractMethod):
    def __init__(self, vocab_size, window_size=3, embedding_size=10, epoch=100) -> None:
        self._model = self._build_model(vocab_size, embedding_size, window_size)
        self.epoch = epoch

    def _build_model(self, vocab_size: int, embedding_size: int, window_size: int):
        context_size = window_size - 1
        inputs = keras.Input(shape=(context_size,))
        x = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)(
            inputs
        )
        x = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
        outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, input, target):
        input = np.array(input)
        target = np.array(target)
        print(input.shape, target.shape)
        history = self._model.fit(input, target, verbose=1, epochs=self.epoch)
        return history.history["loss"]

    def predict(self, input):
        input = np.array(input)
        return self._model.predict(input)

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)


class SkipGram(AbstractMethod):
    def __init__(self, vocab_size, embedding_size=10, epoch=100) -> None:
        self._model = self._build_model(vocab_size, embedding_size)
        self.epoch = epoch

    def _build_model(self, vocab_size: int, embedding_size: int):
        inputs = keras.Input(shape=(vocab_size,))
        x = keras.layers.Dense(embedding_size, activation="linear")(inputs)
        outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, input, target):
        input = np.array(input)  # shape: (3955, vocab_size)
        target = np.array(target)  # shape: (3955, vocab_size)
        print(input.shape, target.shape)
        history = self._model.fit(input, target, verbose=1, epochs=self.epoch)
        return history.history["loss"]

    def predict(self, input):
        input = np.array(input)
        return self._model.predict(input)

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)
