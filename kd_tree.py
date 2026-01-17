# kd_tree.py
import numpy as np
from heapq import heappush, heappop

class KDNode:
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point  
        self.label = label  
        self.axis = axis    
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, X, y):
        self.k = X.shape[1]
        self.root = self._build(X, y, depth=0)


    def _build(self, X, y, depth):
        if len(X) == 0:
            return None
        axis = depth % self.k
        idx = np.argsort(X[:, axis])
        X, y = X[idx], y[idx]
        mid = len(X) // 2
        return KDNode(
            point=X[mid],
            label=y[mid],
            axis=axis,
            left=self._build(X[:mid], y[:mid], depth + 1),
            right=self._build(X[mid + 1 :], y[mid + 1 :], depth + 1),
        )

    @staticmethod
    def _dist(a, b):
        return np.linalg.norm(a - b)

    def knn_search(self, query, k=3):
        heap = []
        self._search(self.root, query, k, heap)
        # Return labels of k nearest neighbors
        return [lbl for (neg_d, lbl) in sorted(heap, key=lambda x: x[0], reverse=True)]

    def _search(self, node, query, k, heap):
        if node is None:
            return
        dist = self._dist(query, node.point)
        if len(heap) < k:
            heappush(heap, (-dist, node.label))
        else:
            if dist < -heap[0][0]:
                heappop(heap)
                heappush(heap, (-dist, node.label))

        axis = node.axis
        diff = query[axis] - node.point[axis]
        close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._search(close, query, k, heap)
        # Check for exploring the other branch
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._search(away, query, k, heap)

    def predict(self, X, k=3):
        preds = []
        for q in X:
            labels = self.knn_search(q, k)
            # majority vote with deterministic tie-break (smallest label)
            values, counts = np.unique(labels, return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.array(preds)
