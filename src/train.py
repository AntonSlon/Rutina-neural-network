from pathlib import Path
import torch.nn
import json
from src.classifier import Classifier
from torch.utils.data import DataLoader, TensorDataset
from src.ruBERT import RuBERT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Train:
    def __init__(self, name):
        self.name = name
        self.model = Classifier()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=0.01)
        self.model.train()
        self.history = {
            "train_acc": [],
            "train_loss": [],
            "test_acc": [],
            "test_loss": []
        }

    def fit(self, train_dataset, test_dataset, epochs: int, batch_size: int):
        x_train = torch.tensor(train_dataset["texts"], dtype=torch.float32)
        y_train = torch.tensor(train_dataset["labels"], dtype=torch.long)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        x_test = torch.tensor(test_dataset["texts"], dtype=torch.float32)
        y_test = torch.tensor(test_dataset["labels"], dtype=torch.long)
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_preds, train_true = [], []

            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                predict = self.model(batch_x)
                loss = self.criterion(predict, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(predict, dim=1)
                train_preds.extend(preds.detach().cpu().numpy())
                train_true.extend(batch_y.detach().cpu().numpy())

            self.model.eval()
            test_loss = 0
            test_preds, test_true = [], []

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    predict = self.model(batch_x)
                    loss_t = self.criterion(predict, batch_y)
                    test_loss += loss_t.item()

                    preds = torch.argmax(predict, dim=1)
                    test_preds.extend(preds.cpu().numpy())
                    test_true.extend(batch_y.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = accuracy_score(train_true, train_preds)

            avg_test_loss = test_loss / len(test_loader)
            avg_test_acc = accuracy_score(test_true, test_preds)

            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(avg_train_acc)
            self.history["test_loss"].append(avg_test_loss)
            self.history["test_acc"].append(avg_test_acc)

            print(f"Epoch {epoch}: "
                  f"Loss (Tr/Te): {avg_train_loss:.4f}/{avg_test_loss:.4f} | "
                  f"Acc (Tr/Te): {avg_train_acc:.4f}/{avg_test_acc:.4f}")
        self._save_test(f"test_{self.name}.pth", x_test, y_test)

    def save_model(self):
        file_path = Path(__file__).parent.parent / "model" / f"classifier_{self.name}.pth"
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_name):
        file_path = Path(__file__).parent.parent / "model" / file_name
        self.model.load_state_dict(torch.load(file_path, weights_only=True))
        return self.model

    def save_history(self):
        file_path = Path(__file__).parent.parent / "model" / f"history_{self.name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f)

    def _save_test(self, file_name, x_test, y_test):
        file_path = Path(__file__).parent.parent / "model" / file_name
        torch.save((x_test, y_test), file_path)


if __name__ == "__main__":
    rubert = RuBERT()
    vectors = rubert.load_vectors("ruBERT_vectors_25000_uncased.npz")

    train_idx, test_idx = train_test_split(
        range(len(vectors["texts"])),
        test_size=0.2,
        random_state=42
    )

    train_vectors = {
        "texts": vectors["texts"][train_idx],
        "labels": vectors["labels"][train_idx]
    }

    test_vectors = {
        "texts": vectors["texts"][test_idx],
        "labels": vectors["labels"][test_idx]
    }

    train = Train("model1")
    train.fit(train_vectors, test_vectors, 10, 128)

    print(train.history)
    train.save_model()
    train.save_history()