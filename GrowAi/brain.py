import json
import os

class Brain:
    def __init__(self, vocab_path='vocab.json'):
        self.vocab_path = vocab_path
        self.vocab = self.load_vocab()

    def load_vocab(self):
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_vocab(self):
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def understand(self, question):
        words = question.lower().split()
        known = [w for w in words if w in self.vocab]
        unknown = [w for w in words if w not in self.vocab]

        response = "ฉันเข้าใจคำว่า: " + ', '.join(known)
        if unknown:
            response += "\nฉันยังไม่รู้จักคำเหล่านี้: " + ', '.join(unknown)
        return response

    def learn(self, word, meaning):
        self.vocab[word.lower()] = meaning
        self.save_vocab()
