import heapq
import random

import numpy as np


class NSW:
    def __init__(self, dim, M=16):
        self.dim = dim
        self.M = M
        self.graph = {}
        self.vectors = []
        self.total_vectors = 0

    def _greedy_search(self, query, entry_point, k):
        visited = {entry_point}
        expanded = set()
        entry_dist = np.linalg.norm(self.vectors[entry_point] - query)
        candidates = [(entry_dist, entry_point)]

        while candidates:
            _, current = heapq.heappop(candidates)
            if current in expanded:
                continue
            expanded.add(current)

            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_dist = np.linalg.norm(self.vectors[neighbor] - query)
                    heapq.heappush(candidates, (neighbor_dist, neighbor))

        distances = [
            (np.linalg.norm(self.vectors[n] - query), n) for n in visited
        ]
        distances.sort()
        return [n for _, n in distances[:k]]

    def insert(self, vecs):
        for vec in vecs:
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

    def search(self, query, k=10):
        if self.total_vectors == 0:
            return []

        entry_point = random.randint(0, self.total_vectors - 1)
        return self._greedy_search(query, entry_point, k)
