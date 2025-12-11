import numpy as np

def euclidean_distance(d1, d2):
    return np.sqrt(np.sum(d1 - d2) ** 2)

def cosine_distance(d1, d2):
    dot_product = np.dot(d1, d2)
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    return 1 -(dot_product / (norm_d1 * norm_d2))

def manhattan_distance(d1, d2):
    return np.sum(np.abs(d1 - d2))