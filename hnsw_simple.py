import heapq
import random

import numpy as np
from sentence_transformers import SentenceTransformer


class HNSWSimple:
    def __init__(self, M=16, Mmax=None, Mmax0=None, efConstruction=200, ef=50, mL=None):
        self.M = M
        self.Mmax = Mmax if Mmax is not None else M
        self.Mmax0 = Mmax0 if Mmax0 is not None else 2 * M
        self.efConstruction = efConstruction
        self.ef = ef
        self.mL = mL if mL is not None else 1.0 / np.log(M)
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        self.graphs = {}
        self.vectors = []
        self.sentences = []
        self.entry_point = None
        self.max_level = 0
        self.total_vectors = 0

    def _search_layer(self, query, entry_point, ef, layer):
        visited = {entry_point}
        entry_dist = np.linalg.norm(self.vectors[entry_point] - query)
        candidates = [(entry_dist, entry_point)]
        W = [(-entry_dist, entry_point)]
        heapq.heapify(W)

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            furthest_neg, _ = W[0]
            furthest_dist = -furthest_neg
            if dist_c > furthest_dist:
                break

            for e in self.graphs.get(c, {}).get(layer, []):
                if e not in visited:
                    visited.add(e)
                    dist_e = np.linalg.norm(self.vectors[e] - query)
                    furthest_neg, _ = W[0]
                    furthest_dist = -furthest_neg
                    if dist_e < furthest_dist or len(W) < ef:
                        heapq.heappush(candidates, (dist_e, e))
                        heapq.heappush(W, (-dist_e, e))
                        if len(W) > ef:
                            heapq.heappop(W)

        return [(-neg_dist, node_id) for neg_dist, node_id in W]

    def _select_neighbors_simple(self, query, candidates, M):
        if not candidates:
            return []
        distances = [(np.linalg.norm(self.vectors[c] - query), c) for c in candidates]
        distances.sort()
        return [c for _, c in distances[:M]]

    def insert(self, sentences):
        vectors = self.model.encode(sentences)
        self.sentences.extend(sentences)

        for vec in vectors:
            node_id = self.total_vectors
            self.vectors.append(vec)
            self.graphs[node_id] = {}

            l = int(-np.log(np.random.uniform(0.0, 1.0)) * self.mL)

            if self.entry_point is None:
                self.entry_point = node_id
                self.max_level = l
                for level in range(l + 1):
                    self.graphs[node_id][level] = []
                self.total_vectors += 1
                continue

            ep = self.entry_point
            L = self.max_level

            for lc in range(L, l, -1):
                W = self._search_layer(vec, ep, 1, lc)
                ep = min(W, key=lambda x: x[0])[1]

            for lc in range(min(L, l), -1, -1):
                W = self._search_layer(vec, ep, self.efConstruction, lc)
                candidates = [n for _, n in W]
                neighbors = self._select_neighbors_simple(vec, candidates, self.M)

                for neighbor in neighbors:
                    if lc not in self.graphs[node_id]:
                        self.graphs[node_id][lc] = []
                    if lc not in self.graphs[neighbor]:
                        self.graphs[neighbor][lc] = []
                    if neighbor not in self.graphs[node_id][lc]:
                        self.graphs[node_id][lc].append(neighbor)
                    if node_id not in self.graphs[neighbor][lc]:
                        self.graphs[neighbor][lc].append(node_id)

                for e in neighbors:
                    e_conn = self.graphs[e].get(lc, [])
                    max_conn = self.Mmax0 if lc == 0 else self.Mmax
                    if len(e_conn) > max_conn:
                        e_new_conn = self._select_neighbors_simple(
                            self.vectors[e], e_conn, max_conn
                        )
                        self.graphs[e][lc] = e_new_conn

                ep = min(W, key=lambda x: x[0])[1]

            if l > self.max_level:
                self.entry_point = node_id
                self.max_level = l

            self.total_vectors += 1

    def search(self, query, k=3):
        if self.total_vectors == 0:
            return []

        query_vector = self.model.encode([query])[0]

        ep = self.entry_point
        L = self.max_level

        for lc in range(L, 0, -1):
            W = self._search_layer(query_vector, ep, 1, lc)
            ep = min(W, key=lambda x: x[0])[1]

        W = self._search_layer(query_vector, ep, self.ef, 0)
        W.sort(key=lambda x: x[0])

        results = []
        for _, node_id in W[:k]:
            dist = np.linalg.norm(self.vectors[node_id] - query_vector)
            results.append((node_id, dist, self.sentences[node_id]))

        return results
