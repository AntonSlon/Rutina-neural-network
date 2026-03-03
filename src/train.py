import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from vectorizer import Vectorizer


class TextClassifier:
    def __init__(self, size_dim, size_classes):
        self.size_dim = size_dim
        self.size_classes = size_classes
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(self.size_dim, )),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(self.size_classes, activation="softmax")
        ])
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def save_model(self, path="models/model.h5"):
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        self.model.save(path)

    def train(self, x, y):
        y_vec = tf.keras.utils.to_categorical(y, num_classes=self.size_classes)
        self.model.fit(x, y_vec, epochs=100, batch_size=2)


if __name__ == "__main__":
    vectorizer = Vectorizer()
    x, y = vectorizer.get_data()
    train = TextClassifier(128, 3)
    train.train(x, y)
    train.save_model()
