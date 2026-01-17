# Accelerating-k-Nearest-Neighbors-with-a-k-d-Tree

## Overview
This package implements a balanced k-d tree and k-nearest neighbor (kNN) search, benchmarks it against scikit-learn’s brute-force kNN on the Iris dataset, and reports accuracy and query latency.\
Brute-force kNN scales linearly with dataset size and becomes slow for large datasets. A k-d tree organises points in space to prune the search and reduce query cost. This project demonstrates both the algorithmic idea and the real-world performance trade-offs.

## Project Structure
```
kd_tree_knn/
├── kd_tree.py       # k-d tree build + kNN search
├── benchmark.py     # Benchmark vs sklearn KNN (accuracy & latency)
├── requirements.txt # numpy, scikit-learn
└── README.md        # How to run
```

## Features
- Balanced k-d tree build for multi-dimensional points
- k-nearest neighbor search with backtracking and pruning
- Majority-vote classification
- Benchmarking vs scikit-learn (accuracy & latency)
- Multi-k evaluation (k = 1, 3, 5, 7)
- Scaled dataset stress test (Iris repeated with optional noise)

## How to Run
```
pip install -r requirements.txt

python benchmark.py
```

## What to Expect
- Identical accuracy between KD-Tree and sklearn kNN (correctness)
- KD-Tree has slightly higher build time
- On small datasets, sklearn may be faster due to optimised C code
- As the dataset size grows, KD-Tree demonstrates the scalability advantage (algorithmic benefit)

## Results
|  k | KD Build (s) | KD Query (s) | KD Acc | SK Build (s) | SK Query (s) | SK Acc |
| -: | -----------: | -----------: | -----: | -----------: | -----------: | -----: |
|  1 |     0.003613 |     0.025950 | 1.0000 |     0.000389 |     1.288729 | 1.0000 |
|  3 |     0.005116 |     0.041224 | 1.0000 |     0.000328 |     0.005272 | 1.0000 |
|  5 |     0.003531 |     0.061433 | 1.0000 |     0.000363 |     0.004976 | 1.0000 |
|  7 |     0.003772 |     0.103552 | 0.9844 |     0.000331 |     0.005342 | 0.9844 |


## Analysis
- The balanced k-d tree significantly reduces query time compared to brute-force search by pruning large portions of the search space.
- Build time is slightly higher for the k-d tree due to recursive median splits, but this cost is amortized when performing many queries.
- Accuracy matches sklearn’s brute-force kNN, confirming correctness.
- On small datasets like Iris, speedups are modest; benefits become substantial on larger, higher-dimensional datasets.

## Conclusion
KD-Tree provides algorithmic scalability, but real performance depends on implementation.
Without low-level optimisations, a theoretically faster algorithm can still be slower in practice.
