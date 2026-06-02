import heapq
import random

import numpy as np
from sentence_transformers import SentenceTransformer


class NSW:
    def __init__(self, M=16):
        self.M = M
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        self.graph = {}
        self.vectors = []
        self.sentences = []
        self.total_vectors = 0

    def _greedy_search(self, query_vector, entry_point, k):
        visited = {entry_point}
        expanded = set()
        entry_dist = np.linalg.norm(self.vectors[entry_point] - query_vector)
        candidates = [(entry_dist, entry_point)]

        while candidates:
            _, current = heapq.heappop(candidates)
            if current in expanded:
                continue
            expanded.add(current)

            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_dist = np.linalg.norm(
                        self.vectors[neighbor] - query_vector
                    )
                    heapq.heappush(candidates, (neighbor_dist, neighbor))

        distances = [
            (np.linalg.norm(self.vectors[n] - query_vector), n) for n in visited
        ]
        distances.sort()
        return [n for _, n in distances[:k]]

    def insert(self, sentences):
        vectors = self.model.encode(sentences)
        self.sentences.extend(sentences)

        for vec in vectors:
            node_id = self.total_vectors
            self.vectors.append(vec)
            self.graph[node_id] = []

            if node_id == 0:
                self.total_vectors += 1
                continue

            entry_point = random.randint(0, node_id - 1)
            neighbors = self._greedy_search(vec, entry_point, self.M)

            for neighbor in neighbors:
                self.graph[neighbor].append(node_id)
                self.graph[node_id].append(neighbor)

            self.total_vectors += 1

    def search(self, query, k=3):
        query_vector = self.model.encode([query])[0]

        if self.total_vectors == 0:
            return []

        entry_point = random.randint(0, self.total_vectors - 1)
        nearest = self._greedy_search(query_vector, entry_point, k)

        results = []
        for node_id in nearest:
            dist = np.linalg.norm(self.vectors[node_id] - query_vector)
            results.append((node_id, dist, self.sentences[node_id]))

        return results