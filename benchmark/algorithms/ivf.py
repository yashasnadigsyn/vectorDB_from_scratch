import numpy as np


class KMeansIVF:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, vectors):
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
    def __init__(self, dim, num_clusters=4, n_probes=1):
        self.dim = dim
        self.num_clusters = num_clusters
        self.n_probes = n_probes
        self.kmeans_ivf = KMeansIVF(k=num_clusters)
        self.inverted_index = {}
        self.centroids = None
        self.vector_store = []
        self.total_vectors = 0
        self.is_trained = False

    def train(self, vecs):
        self.centroids = self.kmeans_ivf.fit(vecs)
        self.is_trained = True

        for i in range(self.num_clusters):
            self.inverted_index[i] = []

        distances = np.array(
            [np.linalg.norm(vecs - c, axis=1) for c in self.centroids]
        ).T
        labels = np.argmin(distances, axis=1)

        for vector_id, vector in enumerate(vecs):
            cluster_id = labels[vector_id]
            self.inverted_index[cluster_id].append(vector_id)
            self.vector_store.append(vector)

        self.total_vectors = len(vecs)

    def add(self, vecs):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")

        distances = np.array(
            [np.linalg.norm(vecs - c, axis=1) for c in self.centroids]
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

        distances_to_centroids = np.linalg.norm(self.centroids - query, axis=1)
        nearest_centroids = np.argsort(distances_to_centroids)[: self.n_probes]

        candidate_ids = []
        for cluster_id in nearest_centroids:
            candidate_ids.extend(self.inverted_index[cluster_id])

        candidate_vecs = np.array([self.vector_store[i] for i in candidate_ids])
        candidate_distances = np.linalg.norm(candidate_vecs - query, axis=1)
        best_indices = np.argsort(candidate_distances)[:k]

        return [candidate_ids[idx] for idx in best_indices]
