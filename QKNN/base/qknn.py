import numpy as np
from base.swaptest import swap_test

class HybridQKNN:
  def __init__(self, k=3, niter=5, shots=4096):
    self.k = k
    self.niter = niter
    self.shots = shots

  def fit(self, Dtrain, Ytrain):
    self.Dtrain = Dtrain
    self.Ytrain = Ytrain

  def quantum_distance(self, v1, v2):
    distance = []
    for _ in range(self.niter):
      _, _, _, dist = swap_test(v1, v2, self.shots)
      distance.append(dist)
    return np.mean(distance)

  def predict_one(self, vi):
    distances = []
    for uj, label in zip(self.Dtrain, self.Ytrain):
      dist = self.quantum_distance(vi, uj)
      distances.append((dist, label))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:self.k]
    count = {}
    for _, label in neighbors:
      count[label] = count.get(label, 0) + 1
    return max(count, key=count.get)

  def predict(self, Dtest):
    predictions = []
    for vi in Dtest:
      predictions.append(self.predict_one(vi))
    return np.array(predictions)