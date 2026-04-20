import numpy as np
import torch
import json
from pathlib import Path
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset


class RuBERT:
    def __init__(self):
        cache_dir = Path(__file__).resolve().parent.parent / "models_cache"
        self._tokenizer = BertTokenizer.from_pretrained(
            'DeepPavlov/rubert-base-cased',
            do_lower_case=True,
            cache_dir=str(cache_dir)
        )
        self.model = BertModel.from_pretrained(
            'DeepPavlov/rubert-base-cased',
            cache_dir=str(cache_dir)
        )
        self.model.eval()

    def tokenize(self, text):
        return self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    def load_corpus(self, file_name):
        file_path = Path(__file__).parent.parent / "data" / file_name
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)
                yield text["text"], text["label"]

    def vectorize(self, corpus):
        vectors = []
        labels = []
        for text, label in corpus:
            labels.append(label)
            tokens = self.tokenize(text)
            with torch.no_grad():
                outputs = self.model(**tokens)
            cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            vectors.append(cls_vector)
        return TensorDataset(torch.tensor(vectors), torch.tensor(labels))

    def save_vectors(self, file_name, vectors: TensorDataset):
        texts, labels = vectors.tensors
        file_path = Path(__file__).parent.parent / "model" / file_name
        np.savez(file_path, texts=texts, labels=labels)

    def load_vectors(self, file_name):
        file_path = Path(__file__).parent.parent / "model" / file_name
        loaded_vectors = np.load(file_path)
        return loaded_vectors


if __name__ == "__main__":
    rubert = RuBERT()
    corpus = list(rubert.load_corpus("dataset.jsonl"))
    vec = rubert.vectorize(corpus)
    rubert.save_vectors("ruBERT_dataset", vec)
