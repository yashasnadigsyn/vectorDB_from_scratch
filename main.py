import time

import numpy as np

from brute_force import BruteForceVectorDB

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

bf_db = BruteForceVectorDB()
bf_db.insert(SENTENCES)
for query in QUERIES:
    results = bf_db.search(query)
    print(f"Query: {query}")
    print(f"Results: {results}")
    print("---")
