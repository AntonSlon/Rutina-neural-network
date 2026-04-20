from pathlib import Path
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import torch
import torch.nn.functional as F
import json
from src.ruBERT import RuBERT
from src.train import Train


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
        self.db = self.load_advices("tips.pt")
        self.cross_encoder = None
        self.ruBERT_model = None
        self.classifier = None

        if load_predict_stack:
            self._ensure_predict_stack()

    def _ensure_predict_stack(self):
        if self.ruBERT_model is None:
            print("Loading RuBERT model...")
            self.ruBERT_model = RuBERT()

        if self.classifier is None:
            print("Loading classifier...")
            self.classifier = Train("model3").load_model("classifier_model3.pth")
            self.classifier.eval()

    def _ensure_cross_encoder(self):
        if self.cross_encoder is None:
            print("Loading cross encoder...")
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

    def predict_with_meta(self, input_text):
        input_text_lower = input_text.lower()
        classifier_vector = self.vectorize_text(input_text_lower)
        category = str(self.get_category(classifier_vector))
        category_vectors = self.db["vectors"][category]
        category_texts = self.db["texts"][category]
        retrieval_vector = self.embed_for_retrieval(input_text_lower)

        similarities = util.cos_sim(retrieval_vector, category_vectors).squeeze(0)
        boosted_similarities = similarities.clone()
        query_words = [word for word in input_text_lower.split() if len(word) >= 3]

        for idx, advice_text in enumerate(category_texts):
            advice_lower = advice_text.lower()
            for word in query_words:
                if word in advice_lower:
                    boosted_similarities[idx] += 0.2
                elif len(word) >= 4 and word[:4] in advice_lower:
                    boosted_similarities[idx] += 0.1

        top_k = min(50, len(category_texts))
        _, indices = boosted_similarities.topk(top_k)
        candidate_texts = [category_texts[idx] for idx in indices.tolist()]

        self._ensure_cross_encoder()
        pairs = [[input_text, text] for text in candidate_texts]
        scores = self.cross_encoder.predict(pairs)

        best_idx = scores.argmax()
        best_text = candidate_texts[best_idx]

        return {
            "text": best_text,
            "category": category,
            "scores": scores,
            "candidate_texts": candidate_texts,
            "similarities": similarities[indices].tolist(),
            "boosted_similarities": boosted_similarities[indices].tolist()
        }

    def give_advice(self, input_text):
        result = self.predict_with_meta(input_text)
        best_category = result["category"]
        best_score = float(result["scores"][result["candidate_texts"].index(result["text"])])

        return result["text"]

    def vectorize_and_save_advices(self, file_name, batch_size=32):
        file_path = Path(__file__).parent.parent / "data" / file_name
        print(f"Loading advice file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            advices = json.load(f)

        db = {
            "texts": {},
            "vectors": {},
            "encoder_name": self.RETRIEVAL_ENCODER_NAME
        }

        for category, sentences in advices.items():
            print(f"Vectorizing category {category}: {len(sentences)} texts")
            db["texts"][category] = sentences
            vectors = self.retrieval_encoder.encode(
                [s.lower() for s in sentences],
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            db["vectors"][category] = vectors.cpu()
            print(f"Category {category} done")

        save_path = Path(__file__).parent.parent / "model" / "tips.pt"
        print(f"Saving advice index to: {save_path}")
        torch.save(db, save_path)
        print("Advice index saved")

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
