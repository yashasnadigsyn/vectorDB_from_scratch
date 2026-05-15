import numpy as np

from flat import BruteForceVectorDB
from ivf import IVFIndex
from ivf_optimized import IVFOptimized
from no_dependency_ivf import IVFIndexPure

SENTENCES = [
    "The king sat proudly on his throne.",
    "A monarch rules the kingdom with wisdom.",
    "The queen wore a golden crown at the palace.",
    "Cats are great companions for lonely people.",
    "Dogs love to play fetch in the park.",
    "The weather today is warm and sunny.",
    "A storm is approaching with heavy rain and thunder.",
    "The royal family attended the coronation ceremony.",
    "Pets can help reduce stress and anxiety in humans.",
    "The temperature will drop significantly tonight with frost.",
    "iPhones are the best smartphones on the market.",
    "Apples are the best fruits on the market.",
    "Samsung phones are better than iphones",
]

QUERIES = [
    "king and royalty",
    "weather forecast",
    "animals and pets",
    "ruler of a country",
    "cold temperatures",
    "apple",
]

# bf_db = BruteForceVectorDB()
# bf_db.insert(SENTENCES)
# for query in QUERIES:
#     results = bf_db.search(query)
#     print(f"Query: {query}")
#     print(f"Results: {results}")
#     print("---")

# ivf_db = IVFIndex()
# ivf_db.train_and_add(SENTENCES)
# for query in QUERIES:
#     results = ivf_db.search(query)
#     print(f"Query: {query}")
#     print(f"Results: {results}")
#     print("---")

# ivf_optimized_db = IVFOptimized()
# ivf_optimized_db.train(SENTENCES)
# ivf_optimized_db.add(SENTENCES)
# for query in QUERIES:
#     results = ivf_optimized_db.search(query)
#     print(f"Query: {query}")
#     print(f"Results: {results}")
#     print("---")

ivf_pure_db = IVFIndexPure()
ivf_pure_db.train(SENTENCES)
ivf_pure_db.add(SENTENCES)
for query in QUERIES:
    results = ivf_pure_db.search(query)
    print(f"Query: {query}")
    print(f"Results: {results}")
    print("---")
