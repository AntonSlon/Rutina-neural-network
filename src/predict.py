from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import json
from src.ruBERT import RuBERT
from src.train import Train


class Predict:
    def __init__(self):
        self.ruBERT_model = RuBERT()
        self.classifier = Train("model3").load_model("classifier_model3.pth")
        self.classifier.eval()
        self.db = self.load_advices()

    def vectorize_text(self, text):
        tokenized = self.ruBERT_model.tokenize(text.lower())
        with torch.no_grad():
            output = self.ruBERT_model.model(**tokenized)
        vectorized = output.last_hidden_state[:, 0, :]
        return vectorized

    def get_category(self, input_vector):
        with torch.no_grad():
            prediction = self.classifier(input_vector)
            probs = F.softmax(prediction, dim=1)
            #print(probs)
            best_class = torch.argmax(probs, dim=1).item()
            return best_class

    def give_advice(self, text):
        input_vector = self.vectorize_text(text)
        category = str(self.get_category(input_vector))
        category_vectors = self.db["vectors"][category]
        category_texts = self.db["texts"][category]
        similarities = F.cosine_similarity(input_vector, category_vectors)
        best_id = torch.argmax(similarities).item()
        return category_texts[best_id]

    def vectorize_and_save_advices(self, file_name, batch_size=32):
        file_path = Path(__file__).parent.parent / "data" / file_name
        with open(file_path, "r", encoding="utf-8") as f:
            advices = json.load(f)

        db = {"texts": {}, "vectors": {}}

        for category, sentences in advices.items():
            db["texts"][category] = sentences
            category_vectors = []
            for i in range(0, len(sentences), batch_size):
                batch_sentences = [s.lower() for s in sentences[i:i + batch_size]]
                with torch.no_grad():
                    tokens = self.ruBERT_model.tokenize(batch_sentences)
                    outputs = self.ruBERT_model.model(**tokens)
                    vectors = outputs.last_hidden_state[:, 0, :].detach().cpu()
                    category_vectors.append(vectors)

            db["vectors"][category] = torch.cat(category_vectors, dim=0)

        save_path = Path(__file__).parent.parent / "model" / "advices_25000.pt"
        torch.save(db, save_path)

    def load_advices(self):
        file_path = Path(__file__).parent.parent / "model" / "advices_25000.pt"
        if file_path.exists():
            return torch.load(file_path, weights_only=False)
        return None


if __name__ == "__main__":
    pred = Predict()
    pred.vectorize_and_save_advices("advices_25000.json")

