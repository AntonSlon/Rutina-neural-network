from gensim.models import Word2Vec
from pathlib import Path


class Vectorizer:
    def __init__(self):
        self.model = None

    def fit(self, texts, vector_size, window, min_count, workers, sg, epochs):
        model = Word2Vec(
            sentences=texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            epochs=epochs
        )
        self.model = model
        return model

    def save_model(self):
        if not self.model:
            self.model.save("model/word2vec.model")

    def load_model(self, file_name: str):
        file_path = Path(__file__).parent.parent / "model" / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"model not found: {file_path}")
        else:
            model = Word2Vec.load(file_path)
            self.model = model
            return model

