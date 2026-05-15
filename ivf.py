import numpy as np
from sentence_transformers import SentenceTransformer


class KMeansIVF:
    def __init__(self, k: int, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, vectors: np.ndarray):
        num_vectors = vectors.shape[0]
        random_indices = np.random.choice(num_vectors, self.k, replace=False)
        self.centroids = vectors[random_indices]

        for _ in range(self.max_iters):
            distances = np.array(
                [np.linalg.norm(vectors - c, axis=1) for c in self.centroids]
            ).T
            labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros_like(self.centroids)

            for i in range(self.k):
                bucket_vectors = vectors[labels == i]
                if len(bucket_vectors) > 0:
                    new_centroids[i] = np.mean(bucket_vectors, axis=0)
                else:
                    new_centroids[i] = self.centroids[i]

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self.centroids


class IVFIndex:
    def __init__(self, num_clusters=4, n_probes=1):
        self.num_clusters = num_clusters
        self.n_probes = n_probes
        self.kmeans_ivf = KMeansIVF(k=num_clusters)
        self.inverted_index = {}
        self.centroids = None
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    def train_and_add(self, sentences: list[str]):
        self.sentences = sentences
        vectors = self.model.encode(sentences)
        self.centroids = self.kmeans_ivf.fit(vectors)

        for i in range(self.num_clusters):
            self.inverted_index[i] = []

        distances = np.array(
            [np.linalg.norm(vectors - c, axis=1) for c in self.centroids]
        ).T
        labels = np.argmin(distances, axis=1)

        for vector_id, vector in enumerate(vectors):
            cluster_id = labels[vector_id]
            self.inverted_index[cluster_id].append((vector_id, vector))

    def search(self, query: str, top_k=2):
        query_vector = self.model.encode(query)
        distances_to_centroids = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_centroids = np.argsort(distances_to_centroids)[: self.n_probes]

        candidates = []
        for cluster_id in nearest_centroids:
            candidates.extend(self.inverted_index[cluster_id])

        candidate_ids = [c[0] for c in candidates]
        candidate_vectors = [c[1] for c in candidates]

        candidate_distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
        best_indices = np.argsort(candidate_distances)[:top_k]

        return [
            (
                candidate_ids[idx],
                candidate_distances[idx],
                self.sentences[candidate_ids[idx]],
            )
            for idx in best_indices
        ]
