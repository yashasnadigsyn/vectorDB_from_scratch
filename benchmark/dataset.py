import numpy as np


def generate_dataset(N, dim, seed=42, num_queries=100):
    np.random.seed(seed)
    vectors = np.random.randn(N, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    query_vecs = np.random.randn(num_queries, dim).astype(np.float32)
    query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_vecs = query_vecs / query_norms

    return vectors, query_vecs
