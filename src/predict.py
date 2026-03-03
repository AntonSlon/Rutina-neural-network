import json

from vectorizer import Vectorizer
import tensorflow as tf
import numpy as np
from pathlib import Path
import random

class Predict:
    def __init__(self):
        self.model = tf.keras.models.load_model('models/model.h5')
        self.vectorizer = Vectorizer()

    def _get_predict(self, text: str):
        tokens = self.vectorizer.preprocess_text(text)
        transformed_text = self.vectorizer.transform([tokens])
        predict = self.model.predict(transformed_text)
        print(type(predict))
        result = np.max(self.model.predict(transformed_text))
        return int(np.where(predict == result)[1][0])

    def give_advice(self, text: str, file="advices.json"):
        base_path = Path(__file__).parent.parent
        full_path = base_path / "data" / file
        with open(full_path, "r", encoding="utf-8") as file:
            index = self._get_predict(text)
            data = json.load(file)[str(index)]
            return random.choice(data)


p = Predict()
print(p.give_advice("бухать каждый день"))