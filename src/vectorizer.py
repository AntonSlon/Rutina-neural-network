import re
import json
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from pathlib import Path
import numpy as np


class Vectorizer:
    def __init__(self):
        self.train_data = self._load_data("train_data.json")
        self.texts = [self.preprocess_text(data["text"]) for data in self.train_data]
        self.model = self._fit(self.texts, 128, 5, 1, 4, 1, 10)

    def _load_data(self, filename: str) -> dict:
        base_path = Path(__file__).parent.parent
        full_path = base_path / "data" / filename
        with open(full_path, "r", encoding="utf-8") as file:
            return json.loads(file.read())

    def preprocess_text(self, text: str):
        text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
        tokens = simple_preprocess(text, deacc=True)
        return tokens

    def _fit(self, sentences: list, vector_size: int, window: int, min_count: int, workers: int, sg: int, epochs: int):
        return Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            epochs=epochs
        )

    def transform(self, texts):
        average_vectors = []
        for token in texts:
            vector = [self.model.wv[word] for word in token if word in self.model.wv]
            if len(vector) > 0:
                average_vector = np.mean(vector, axis=0)
            else:
                average_vector = np.zeros(self.model.vector_size)
            average_vectors.append(average_vector)
        return np.array(average_vectors)

    def get_data(self):
        labels = [data["label"] for data in self.train_data]
        x = self.transform(self.texts)
        y = np.array(labels)
        return x, y

