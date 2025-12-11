import numpy as np
from base.distance import euclidean_distance

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, Dtrain, Ytrain):
        self.Dtrain = Dtrain
        self.Ytrain = Ytrain

    def predict_one(self, dj):
        distances = []

        for di, label in zip(self.Dtrain, self.Ytrain):
            dist_ij = euclidean_distance(dj, di)
            distances.append((dist_ij, label))

        distances.sort(key=lambda x: x[0])

        neighbors = distances[:self.k]

        count = {}
        for _, label in neighbors:
            count[label] = count.get(label, 0) + 1

        return max(count, key=count.get)
    
    def predict(self, Dtest):
        predictions = []
        for dj in Dtest:
            predictions.append(self.predict_one(dj))
        return np.array(predictions)