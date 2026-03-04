from src.vectorizer import Vectorizer
import tensorflow as tf
import numpy as np
import random


class Predict:
    def __init__(self):
        self.model = tf.keras.models.load_model('src/models/model.h5')
        self.vectorizer = Vectorizer()

    def _calculate_cos(self, text):
        tokens = self.vectorizer.preprocess_text(text)
        habit_text = self.vectorizer.transform([tokens])
        predict = self.model.predict(habit_text)
        result = np.max(self.model.predict(habit_text))

        key = int(np.where(predict == result)[1][0])
        train_data = self.vectorizer.load_data("advices.json")
        data = train_data[str(key)]
        token = [self.vectorizer.preprocess_text(advice) for advice in data]

        advices_text = self.vectorizer.transform(token)
        advice_norm = np.linalg.norm(advices_text, axis=1)

        habit_norm = np.linalg.norm(habit_text, axis=1)
        cos_sim = np.nan_to_num((habit_text @ advices_text.T) / (advice_norm * habit_norm), nan=-1)
        return cos_sim, data

    def give_advice(self, text: str):
        index, data = self._calculate_cos(text)
        if np.argmax(index) < 0.1:
            return f"Я не уверен на 100%, что правильно тебя понял, но вот базовый совет: {random.choice(data)}"
        else:
            index = np.argmax(index)
            return data[index]


p = Predict()
print(p.give_advice("программированию"))