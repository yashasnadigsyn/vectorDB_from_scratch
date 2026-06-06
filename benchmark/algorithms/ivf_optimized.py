import heapq

import numpy as np


class KMeansPPIVF:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def _kmeans_plus_plus(self, vectors):
        num_vectors = vectors.shape[0]
        first_idx = np.random.choice(num_vectors)
        centroids = [vectors[first_idx]]

        for _ in range(1, self.k):
            distances = np.array(
                [np.linalg.norm(vectors - c, axis=1) for c in centroids]
            ).T
            min_distances = np.min(distances, axis=1)
            probs = min_distances**2
            probs = probs / np.sum(probs)
            next_centroid_idx = np.random.choice(num_vectors, p=probs)
            centroids.append(vectors[next_centroid_idx])

        return np.array(centroids)

    def fit(self, vectors):
        num_vectors = vectors.shape[0]
        self.centroids = self._kmeans_plus_plus(vectors)

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
                    random_idx = np.random.choice(num_vectors)
                    new_centroids[i] = vectors[random_idx]

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self.centroids


class IVFOptimized:
    def __init__(self, dim, num_clusters=4, n_probes=1):
        self.dim = dim
        self.num_clusters = num_clusters
        self.n_probes = n_probes
        self.kmeanspp_ivf = KMeansPPIVF(k=num_clusters)
        self.inverted_index = {i: [] for i in range(num_clusters)}
        self.vector_store = []
        self.is_trained = False
        self.total_vectors = 0

    def train(self, vecs):
        self.kmeanspp_ivf.fit(vecs)
        self.is_trained = True

        distances = np.array(
            [np.linalg.norm(vecs - c, axis=1) for c in self.kmeanspp_ivf.centroids]
        ).T
        labels = np.argmin(distances, axis=1)

        for i, vector in enumerate(vecs):
            cluster_id = labels[i]
            self.inverted_index[cluster_id].append(i)
            self.vector_store.append(vector)

        self.total_vectors = len(vecs)

    def add(self, vecs):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")

        distances = np.array(
            [np.linalg.norm(vecs - c, axis=1) for c in self.kmeanspp_ivf.centroids]
        ).T
        labels = np.argmin(distances, axis=1)

        for i, vector in enumerate(vecs):
            cluster_id = labels[i]
            vector_id = self.total_vectors + i
            self.inverted_index[cluster_id].append(vector_id)
            self.vector_store.append(vector)

        self.total_vectors += len(vecs)

    def search(self, query, k=10):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")

        distances_to_centroids = np.linalg.norm(
            self.kmeanspp_ivf.centroids - query, axis=1
        )
        nearest_centroids = np.argsort(distances_to_centroids)[: self.n_probes]

        candidates = []
        for cluster_id in nearest_centroids:
            candidates.extend(self.inverted_index[cluster_id])

        candidate_distances = []
        for vector_id in candidates:
            dist = np.linalg.norm(self.vector_store[vector_id] - query)
            candidate_distances.append((vector_id, dist))

        best_matches = heapq.nsmallest(k, candidate_distances, key=lambda x: x[1])
        return [match[0] for match in best_matches]
