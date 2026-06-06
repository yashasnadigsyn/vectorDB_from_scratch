import time

import numpy as np


def compute_recall(ground_truth, predicted, k=10):
    scores = []
    for gt, pred in zip(ground_truth, predicted):
        gt_set = set(gt[:k])
        pred_set = set(pred[:k])
        intersection = len(gt_set & pred_set)
        scores.append(intersection / k)
    return scores


def measure_build_time(build_fn):
    start = time.perf_counter()
    build_fn()
    end = time.perf_counter()
    return end - start


def measure_insert_time(insert_fn, vec, num_repeats=100):
    start = time.perf_counter()
    for _ in range(num_repeats):
        insert_fn(vec)
    end = time.perf_counter()
    return (end - start) / num_repeats


def measure_search_qps(search_fn, query, num_repeats=500):
    start = time.perf_counter()
    for _ in range(num_repeats):
        search_fn(query)
    end = time.perf_counter()
    total = end - start
    return num_repeats / total
