import torch
from sklearn.metrics import classification_report
from pathlib import Path
from src.classifier import Classifier
from src.ruBERT import RuBERT
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def get_classification_report(model_name, test_data_name):

    path = Path(__file__).resolve().parent.parent.parent
    model_path = path / "model" / f"{model_name}"

    test_data_path = path / "model" / f"{test_data_name}"
    data_test = torch.load(test_data_path)
    X_test, y_test = data_test[0], data_test[1]

    model = Classifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        best_class = torch.argmax(probs, dim=1)

    print(classification_report(y_test, best_class))


if __name__ == "__main__":
    rubert = RuBERT()
    get_classification_report("classifier_model1.pth", "test_model1.pth")


