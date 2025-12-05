# src/datasets.py

def get_small_dataset():
    return {
        "name": "small_dataset",
        "num_rows": 10_000,
        "num_features": 20,
    }

def get_medium_dataset():
    return {
        "name": "medium_dataset",
        "num_rows": 100_000,
        "num_features": 30,
    }

def get_large_dataset():
    return {
        "name": "large_dataset",
        "num_rows": 1_000_000,
        "num_features": 60,
    }
