import re
import json
import pymorphy3
from gensim.utils import simple_preprocess
import nltk
from pathlib import Path
from nltk.corpus import stopwords


class Preprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("russian"))
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words("russian"))
        self.morph = pymorphy3.MorphAnalyzer()

    def _preprocess(self, text: str):
        words = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
        tokens = simple_preprocess(words, deacc=True)
        normal_tokens = []
        for word in tokens:
            if word not in self.stop_words:
                normal_form = self.morph.parse(word)[0].normal_form
                normal_tokens.append(normal_form)
        return normal_tokens

    def load_corpus(self, file_name: str):
        file_path = Path(__file__).parent.parent / "data" / file_name
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                yield self._preprocess(text)

