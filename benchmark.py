# benchmark.py
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from kd_tree import KDTree


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    # expand dataset
    X = np.vstack([X] * 10)
    y = np.hstack([y] * 10)
    # add noise to reduce duplicates
    X = X + np.random.normal(0, 0.01, X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(22*"="," Benchmark Results for Multiple k Values",22*"=")
    print(f"{'k':>3} | {'KD Build(s)':10} | {'KD Query(s)':10} | {'KD Acc':10} || {'SK Build(s)':10} | {'SK Query(s)':10} | {'SK Acc':10}")
    print("-" * 86)

    for k in [1, 3, 5, 7]:
        # KD-Tree KNN
        t0 = time.perf_counter()
        tree = KDTree(X_train, y_train)
        build_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        kd_preds = tree.predict(X_test, k=k)
        kd_time = time.perf_counter() - t1
        kd_acc = accuracy_score(y_test, kd_preds)

        # sklearn KNN (brute force)
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
        t2 = time.perf_counter()
        knn.fit(X_train, y_train)
        sk_build = time.perf_counter() - t2

        t3 = time.perf_counter()
        sk_preds = knn.predict(X_test)
        sk_time = time.perf_counter() - t3
        sk_acc = accuracy_score(y_test, sk_preds)

        print(f"{k:>3} | {build_time:11.6f} | {kd_time:11.6f} | {kd_acc:10.4f} || {sk_build:11.6f} | {sk_time:11.6f} | {sk_acc:10.4f}")
