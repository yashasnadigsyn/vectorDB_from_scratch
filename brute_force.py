import numpy as np
from sentence_transformers import SentenceTransformer


class BruteForceVectorDB:
    def __init__(self):
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        self.corpus_embs = None
        self.sentences = []

    def insert(self, sentences: list[str]):
        self.sentences.extend(sentences)
        new_embs = self.model.encode(sentences, normalize_embeddings=True)

        if self.corpus_embs is None:
            self.corpus_embs = new_embs
        else:
            self.corpus_embs = np.vstack([self.corpus_embs, new_embs])

    def search(self, query: str, k: int = 3):
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]

        if self.corpus_embs is None:
            return []

        scores = np.dot(self.corpus_embs, query_emb.T)
        top_k_indices = np.argsort(scores, axis=0)[-k:][::-1]

        return [self.sentences[i] for i in top_k_indices]
