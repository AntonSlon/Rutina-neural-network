from pathlib import Path

import pymorphy3
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import torch
import torch.nn.functional as F
import json
from src.ruBERT import RuBERT
from src.train import Train
from rank_bm25 import BM25Okapi


class Predict:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CACHE_DIR = PROJECT_ROOT / "models_cache"
    RETRIEVAL_ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CROSS_ENCODER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    def __init__(self, load_predict_stack=True):
        self.retrieval_encoder = SentenceTransformer(
            self.RETRIEVAL_ENCODER_NAME,
            cache_folder=str(self.CACHE_DIR)
        )
        self.db = self.load_advices("tips_ru.pt")
        self.cross_encoder = None
        self.ruBERT_model = None
        self.classifier = None
        self.morph = pymorphy3.MorphAnalyzer()

        if load_predict_stack:
            self._ensure_predict_stack()

    def _ensure_predict_stack(self):
        if self.ruBERT_model is None:
            self.ruBERT_model = RuBERT()

        if self.classifier is None:
            self.classifier = Train("model3").load_model("classifier_model3.pth")
            self.classifier.eval()

    def _ensure_cross_encoder(self):
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder(
                self.CROSS_ENCODER_NAME,
                cache_folder=str(self.CACHE_DIR)
            )

    def vectorize_text(self, text):
        self._ensure_predict_stack()
        tokenized = self.ruBERT_model.tokenize(text.lower())
        with torch.no_grad():
            output = self.ruBERT_model.model(**tokenized)
        vectorized = output.last_hidden_state[:, 0, :]
        return vectorized

    def get_category(self, input_vector):
        self._ensure_predict_stack()
        with torch.no_grad():
            prediction = self.classifier(input_vector)
        if self.db is None:
            raise FileNotFoundError(
                "Advice index not found. Run vectorize_and_save_advices first."
            )
        probs = F.softmax(prediction, dim=1)
        return torch.argmax(probs, dim=1).item()

    def embed_for_retrieval(self, text):
        embedding = self.retrieval_encoder.encode(
            text.lower(),
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embedding.unsqueeze(0)

    def tokenize_ru(self, text):
        text = ''.join(char if char.isalpha() or char.isspace() or char.isdigit() else ' ' for char in text)
        words = text.lower().split()
        return [self.morph.parse(word)[0].normal_form for word in words if word]

    def give_advice(self, input_text):
        input_text_lower = input_text.lower()
        input_vector = self.vectorize_text(input_text_lower)
        category = str(self.get_category(input_vector))
        category_vectors = self.db["vectors"][category]
        category_texts = self.db["texts"][category]
        category_tokens = self.db["bm25_tokens"][category]
        retrieval_vector = self.embed_for_retrieval(input_text_lower)

        similarities = util.cos_sim(retrieval_vector, category_vectors).squeeze(0)
        values, indices = similarities.topk(50)
        top_indices = indices.tolist()
        candidate_texts = [category_texts[idx] for idx in top_indices]
        candidate_tokens = [category_tokens[idx] for idx in top_indices]
        bm25 = BM25Okapi(candidate_tokens)
        query_tokens = self.tokenize_ru(input_text)
        scores = bm25.get_scores(query_tokens)
        best_scores = {}
        for i, score in enumerate(scores):
            best_scores[i] = score
        max_score = max(best_scores.items(), key=lambda item: item[1])[0]
        return candidate_texts[max_score]

    def vectorize_and_save_advices(self, file_name, batch_size=32):
        file_path = Path(__file__).parent.parent / "data" / file_name
        with open(file_path, "r", encoding="utf-8") as f:
            advices = json.load(f)

        db = {
            "texts": {},
            "bm25_tokens": {},
            "vectors": {},
            "encoder_name": self.RETRIEVAL_ENCODER_NAME
        }

        for category, sentences in advices.items():
            db["texts"][category] = sentences
            db["bm25_tokens"][category] = [self.tokenize_ru(sentence) for sentence in sentences]
            vectors = self.retrieval_encoder.encode(
                [s.lower() for s in sentences],
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            db["vectors"][category] = vectors.cpu()

        save_path = Path(__file__).parent.parent / "model" / "tips_ru.pt"
        torch.save(db, save_path)

    def load_advices(self, file_name):
        file_path = Path(__file__).parent.parent / "model" / file_name
        if file_path.exists():
            db = torch.load(file_path, weights_only=False)
            if db.get("encoder_name") != self.RETRIEVAL_ENCODER_NAME:
                return None
            return db
        return None


if __name__ == "__main__":
    pred = Predict(load_predict_stack=False)
    pred.vectorize_and_save_advices("tips.json")
