import numpy as np


class BruteForceVectorDB:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = None
        self.total_vectors = 0

    def insert(self, vecs):
        if self.vectors is None:
            self.vectors = vecs.copy()
        else:
            self.vectors = np.vstack([self.vectors, vecs])
        self.total_vectors = len(self.vectors)

    def search(self, query, k=10):
        if self.vectors is None:
            return []
        scores = np.dot(self.vectors, query.T)
        top_k_indices = np.argsort(scores, axis=0)[-k:][::-1].flatten()
        return top_k_indices.tolist()
