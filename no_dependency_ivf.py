import heapq
import math
import random
from collections import defaultdict

from sentence_transformers import SentenceTransformer


def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(a**2 for a in v2))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


class KMeansPure:
    def __init__(self, k: int, max_iters=10):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []

    def fit(self, vectors: list[list[float]]):
        self.centroids = random.sample(vectors, self.k)

        for _ in range(self.max_iters):
            labels = defaultdict(list)

            for vector in vectors:
                max_sim = float("-inf")
                best_centroid_idx = 0
                for i, c in enumerate(self.centroids):
                    sim = cosine_similarity(vector, c)
                    if sim > max_sim:
                        max_sim = sim
                        best_centroid_idx = i
                labels[best_centroid_idx].append(vector)

            new_centroids = []
            for i in range(self.k):
                if labels[i]:
                    mean_vec = [sum(dim) / len(labels[i]) for dim in zip(*labels[i])]
                    new_centroids.append(mean_vec)
                else:
                    new_centroids.append(self.centroids[i])

            if self.centroids == new_centroids:
                break

            self.centroids = new_centroids

        return self.centroids


class IVFIndexPure:
    def __init__(self, num_clusters=4, n_probes=2):
        self.num_clusters = num_clusters
        self.n_probes = n_probes
        self.kmeans = KMeansPure(k=num_clusters)
        self.inverted_index = defaultdict(list)
        self.is_trained = False
        self.sentences = []
        self.total_vectors = 0
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    def train(self, sentences: list[str]):
        self.sentences = list(sentences)
        vectors = self.model.encode(sentences).tolist()
        self.kmeans.fit(vectors)
        self.is_trained = True

    def add(self, sentences: list[str]):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")

        vectors = self.model.encode(sentences).tolist()
        self.sentences.extend(sentences)

        for i, vector in enumerate(vectors):
            max_sim = float("-inf")
            best_cluster_id = 0
            for cluster_id, c in enumerate(self.kmeans.centroids):
                sim = cosine_similarity(c, vector)
                if sim > max_sim:
                    max_sim = sim
                    best_cluster_id = cluster_id

            vector_id = self.total_vectors + i
            self.inverted_index[best_cluster_id].append((vector_id, vector))

        self.total_vectors += len(vectors)

    def search(self, query: str, top_k=3):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")

        query_vector = self.model.encode([query]).tolist()[0]
        centroid_similarities = []
        for cluster_id, c in enumerate(self.kmeans.centroids):
            sim = cosine_similarity(c, query_vector)
            centroid_similarities.append((cluster_id, sim))

        nearest_centroids = sorted(
            centroid_similarities, key=lambda x: x[1], reverse=True
        )[: self.n_probes]

        candidates = []
        for cluster_id, _ in nearest_centroids:
            candidates.extend(self.inverted_index[cluster_id])

        candidate_similarities = []
        for vector_id, vector in candidates:
            sim = cosine_similarity(vector, query_vector)
            candidate_similarities.append((vector_id, sim))

        best_matches = heapq.nlargest(top_k, candidate_similarities, key=lambda x: x[1])

        return [
            (match[0], match[1], self.sentences[match[0]]) for match in best_matches
        ]
