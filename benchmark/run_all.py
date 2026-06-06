import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.algorithms.flat import BruteForceVectorDB
from benchmark.algorithms.hnsw_heuristic import HNSWHeuristic
from benchmark.algorithms.hnsw_simple import HNSWSimple
from benchmark.algorithms.ivf import IVFIndex
from benchmark.algorithms.ivf_optimized import IVFOptimized
from benchmark.algorithms.nsw import NSW
from benchmark.dataset import generate_dataset
from benchmark.metrics import compute_recall

K = 10
NUM_QUERIES = 100


def run_scaling_benchmark(out_path):
    dim = 128
    sizes = [1000, 5000, 10000]
    results = []

    for N in sizes:
        print(f"\n=== Scaling: N={N}, dim={dim} ===")
        vecs, queries = generate_dataset(N, dim, num_queries=NUM_QUERIES)

        ground_truth = []
        flat_db = BruteForceVectorDB(dim)
        flat_db.insert(vecs)
        for i in range(NUM_QUERIES):
            gt = flat_db.search(queries[i], k=K)
            ground_truth.append(gt)

        row = {"N": N, "dim": dim}

        t0 = time.perf_counter()
        flat_db2 = BruteForceVectorDB(dim)
        flat_db2.insert(vecs)
        row["flat_build_time"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            flat_db2.search(queries[i], k=K)
        flat_total = time.perf_counter() - t0
        row["flat_qps"] = NUM_QUERIES / flat_total
        row["flat_recall"] = 1.0
        print(f"  Flat: build={row['flat_build_time']:.3f}s, qps={row['flat_qps']:.1f}")

        ivf = IVFIndex(dim, num_clusters=int(np.sqrt(N)), n_probes=4)
        t0 = time.perf_counter()
        ivf.train(vecs)
        row["ivf_build_time"] = time.perf_counter() - t0
        ivf_preds = []
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            pred = ivf.search(queries[i], k=K)
            ivf_preds.append(pred)
        ivf_total = time.perf_counter() - t0
        ivf_recalls = compute_recall(ground_truth, ivf_preds, k=K)
        row["ivf_qps"] = NUM_QUERIES / ivf_total
        row["ivf_recall_avg"] = float(np.mean(ivf_recalls))
        row["ivf_recall_std"] = float(np.std(ivf_recalls))
        print(
            f"  IVF: build={row['ivf_build_time']:.3f}s, qps={row['ivf_qps']:.1f}, recall={row['ivf_recall_avg']:.4f}"
        )

        ivfopt = IVFOptimized(dim, num_clusters=int(np.sqrt(N)), n_probes=4)
        t0 = time.perf_counter()
        ivfopt.train(vecs)
        row["ivfopt_build_time"] = time.perf_counter() - t0
        ivfopt_preds = []
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            pred = ivfopt.search(queries[i], k=K)
            ivfopt_preds.append(pred)
        ivfopt_total = time.perf_counter() - t0
        ivfopt_recalls = compute_recall(ground_truth, ivfopt_preds, k=K)
        row["ivfopt_qps"] = NUM_QUERIES / ivfopt_total
        row["ivfopt_recall_avg"] = float(np.mean(ivfopt_recalls))
        row["ivfopt_recall_std"] = float(np.std(ivfopt_recalls))
        print(
            f"  IVF-Opt: build={row['ivfopt_build_time']:.3f}s, qps={row['ivfopt_qps']:.1f}, recall={row['ivfopt_recall_avg']:.4f}"
        )

        nsw = NSW(dim, M=16)
        t0 = time.perf_counter()
        nsw.insert(vecs)
        row["nsw_build_time"] = time.perf_counter() - t0
        nsw_preds = []
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            pred = nsw.search(queries[i], k=K)
            nsw_preds.append(pred)
        nsw_total = time.perf_counter() - t0
        nsw_recalls = compute_recall(ground_truth, nsw_preds, k=K)
        row["nsw_qps"] = NUM_QUERIES / nsw_total
        row["nsw_recall_avg"] = float(np.mean(nsw_recalls))
        row["nsw_recall_std"] = float(np.std(nsw_recalls))
        print(
            f"  NSW: build={row['nsw_build_time']:.3f}s, qps={row['nsw_qps']:.1f}, recall={row['nsw_recall_avg']:.4f}"
        )

        hnsw = HNSWSimple(dim, M=16, efConstruction=200, ef=50)
        t0 = time.perf_counter()
        hnsw.insert(vecs)
        row["hnsw_build_time"] = time.perf_counter() - t0
        hnsw_preds = []
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            pred = hnsw.search(queries[i], k=K)
            hnsw_preds.append(pred)
        hnsw_total = time.perf_counter() - t0
        hnsw_recalls = compute_recall(ground_truth, hnsw_preds, k=K)
        row["hnsw_qps"] = NUM_QUERIES / hnsw_total
        row["hnsw_recall_avg"] = float(np.mean(hnsw_recalls))
        row["hnsw_recall_std"] = float(np.std(hnsw_recalls))
        print(
            f"  HNSW: build={row['hnsw_build_time']:.3f}s, qps={row['hnsw_qps']:.1f}, recall={row['hnsw_recall_avg']:.4f}"
        )

        results.append(row)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def run_tuning_benchmark(out_path):
    N = 100000
    dim = 128
    print(f"\n=== Tuning: N={N}, dim={dim} ===")
    vecs, queries = generate_dataset(N, dim, num_queries=NUM_QUERIES)

    flat_db = BruteForceVectorDB(dim)
    flat_db.insert(vecs)
    ground_truth = []
    for i in range(NUM_QUERIES):
        gt = flat_db.search(queries[i], k=K)
        ground_truth.append(gt)

    results = []

    ivf_probes = [1, 2, 4, 8, 16]
    for n_probes in ivf_probes:
        ivfopt = IVFOptimized(dim, num_clusters=int(np.sqrt(N)), n_probes=n_probes)
        ivfopt.train(vecs)
        ivfopt_preds = []
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            pred = ivfopt.search(queries[i], k=K)
            ivfopt_preds.append(pred)
        total = time.perf_counter() - t0
        ivfopt_recalls = compute_recall(ground_truth, ivfopt_preds, k=K)
        results.append(
            {
                "algo": "IVF",
                "param": "n_probes",
                "param_value": n_probes,
                "qps": NUM_QUERIES / total,
                "recall_avg": float(np.mean(ivfopt_recalls)),
                "recall_std": float(np.std(ivfopt_recalls)),
            }
        )
        print(
            f"  IVF n_probes={n_probes}: qps={results[-1]['qps']:.1f}, recall={results[-1]['recall_avg']:.4f}"
        )

    hnsw_efs = [5, 10, 20, 50, 100, 200]
    for ef in hnsw_efs:
        hnsw = HNSWSimple(dim, M=16, efConstruction=200, ef=ef)
        hnsw.insert(vecs)
        hnsw_preds = []
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            pred = hnsw.search(queries[i], k=K)
            hnsw_preds.append(pred)
        total = time.perf_counter() - t0
        hnsw_recalls = compute_recall(ground_truth, hnsw_preds, k=K)
        results.append(
            {
                "algo": "HNSW",
                "param": "ef",
                "param_value": ef,
                "qps": NUM_QUERIES / total,
                "recall_avg": float(np.mean(hnsw_recalls)),
                "recall_std": float(np.std(hnsw_recalls)),
            }
        )
        print(
            f"  HNSW ef={ef}: qps={results[-1]['qps']:.1f}, recall={results[-1]['recall_avg']:.4f}"
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def run_dimensionality_benchmark(out_path):
    N = 50000
    dims = [64, 128, 256, 512, 768]
    results = []

    for dim in dims:
        print(f"\n=== Dimensionality: N={N}, dim={dim} ===")
        vecs, queries = generate_dataset(N, dim, num_queries=NUM_QUERIES)

        flat_db = BruteForceVectorDB(dim)
        flat_db.insert(vecs)
        ground_truth = []
        for i in range(NUM_QUERIES):
            gt = flat_db.search(queries[i], k=K)
            ground_truth.append(gt)

        row = {"N": N, "dim": dim}

        ivfopt = IVFOptimized(dim, num_clusters=int(np.sqrt(N)), n_probes=4)
        t0 = time.perf_counter()
        ivfopt.train(vecs)
        row["ivf_build_time"] = time.perf_counter() - t0
        ivfopt_preds = []
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            pred = ivfopt.search(queries[i], k=K)
            ivfopt_preds.append(pred)
        ivfopt_total = time.perf_counter() - t0
        ivfopt_recalls = compute_recall(ground_truth, ivfopt_preds, k=K)
        row["ivf_qps"] = NUM_QUERIES / ivfopt_total
        row["ivf_recall_avg"] = float(np.mean(ivfopt_recalls))
        row["ivf_recall_std"] = float(np.std(ivfopt_recalls))

        hnsw = HNSWSimple(dim, M=16, efConstruction=200, ef=50)
        t0 = time.perf_counter()
        hnsw.insert(vecs)
        row["hnsw_build_time"] = time.perf_counter() - t0
        hnsw_preds = []
        t0 = time.perf_counter()
        for i in range(NUM_QUERIES):
            pred = hnsw.search(queries[i], k=K)
            hnsw_preds.append(pred)
        hnsw_total = time.perf_counter() - t0
        hnsw_recalls = compute_recall(ground_truth, hnsw_preds, k=K)
        row["hnsw_qps"] = NUM_QUERIES / hnsw_total
        row["hnsw_recall_avg"] = float(np.mean(hnsw_recalls))
        row["hnsw_recall_std"] = float(np.std(hnsw_recalls))

        print(
            f"  IVF: build={row['ivf_build_time']:.3f}s, qps={row['ivf_qps']:.1f}, recall={row['ivf_recall_avg']:.4f}"
        )
        print(
            f"  HNSW: build={row['hnsw_build_time']:.3f}s, qps={row['hnsw_qps']:.1f}, recall={row['hnsw_recall_avg']:.4f}"
        )

        results.append(row)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def run_insert_benchmark(out_path):
    dim = 128
    sizes = [1000, 5000, 10000, 50000, 100000, 200000, 300000]
    results = []

    for N in sizes:
        print(f"\n=== Insertion: N={N}, dim={dim} ===")
        vecs, queries = generate_dataset(N, dim, num_queries=1)

        row = {"N": N, "dim": dim}

        t0 = time.perf_counter()
        flat_db = BruteForceVectorDB(dim)
        flat_db.insert(vecs)
        row["flat_total_insert"] = time.perf_counter() - t0
        row["flat_per_vector"] = row["flat_total_insert"] / N

        t0 = time.perf_counter()
        ivfopt = IVFOptimized(dim, num_clusters=int(np.sqrt(N)), n_probes=4)
        ivfopt.train(vecs[: max(N // 2, 1000)])
        ivfopt.add(vecs[max(N // 2, 1000) :])
        row["ivf_total_insert"] = time.perf_counter() - t0
        row["ivf_per_vector"] = row["ivf_total_insert"] / N

        t0 = time.perf_counter()
        nsw = NSW(dim, M=16)
        nsw.insert(vecs)
        row["nsw_total_insert"] = time.perf_counter() - t0
        row["nsw_per_vector"] = row["nsw_total_insert"] / N

        t0 = time.perf_counter()
        hnsw = HNSWSimple(dim, M=16, efConstruction=200, ef=50)
        hnsw.insert(vecs)
        row["hnsw_total_insert"] = time.perf_counter() - t0
        row["hnsw_per_vector"] = row["hnsw_total_insert"] / N

        print(
            f"  Flat: {row['flat_total_insert']:.3f}s ({row['flat_per_vector'] * 1e6:.1f}μs/vec)"
        )
        print(
            f"  IVF: {row['ivf_total_insert']:.3f}s ({row['ivf_per_vector'] * 1e6:.1f}μs/vec)"
        )
        print(
            f"  NSW: {row['nsw_total_insert']:.3f}s ({row['nsw_per_vector'] * 1e6:.1f}μs/vec)"
        )
        print(
            f"  HNSW: {row['hnsw_total_insert']:.3f}s ({row['hnsw_per_vector'] * 1e6:.1f}μs/vec)"
        )

        results.append(row)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("BENCHMARK 1: Scaling (recall, qps, build_time vs N)")
    print("=" * 60)
    run_scaling_benchmark(os.path.join(out_dir, "scaling.json"))

    print("\n" + "=" * 60)
    print("BENCHMARK 2: Tuning knobs (recall vs qps Pareto)")
    print("=" * 60)
    run_tuning_benchmark(os.path.join(out_dir, "tuning.json"))

    print("\n" + "=" * 60)
    print("BENCHMARK 3: Dimensionality (recall vs dim)")
    print("=" * 60)
    run_dimensionality_benchmark(os.path.join(out_dir, "dimensionality.json"))

    print("\n" + "=" * 60)
    print("BENCHMARK 4: Insertion time (total and per-vector)")
    print("=" * 60)
    run_insert_benchmark(os.path.join(out_dir, "insertion.json"))
