# Short Report: Performance Analysis

## Experimental Setup
- Dataset: Iris (150 samples, 4 features, 3 classes), scaled to 1,500 samples  
- Train/Test split: 70/30 (stratified)  
- k values: 1, 3, 5, 7  
- Metrics: Accuracy, Build Time, Query Time  

## Results

|  k | KD Build (s) | KD Query (s) | KD Acc | SK Build (s) | SK Query (s) | SK Acc |
| -: | -----------: | -----------: | -----: | -----------: | -----------: | -----: |
|  1 |     0.003613 |     0.025950 | 1.0000 |     0.000389 |     1.288729 | 1.0000 |
|  3 |     0.005116 |     0.041224 | 1.0000 |     0.000328 |     0.005272 | 1.0000 |
|  5 |     0.003531 |     0.061433 | 1.0000 |     0.000363 |     0.004976 | 1.0000 |
|  7 |     0.003772 |     0.103552 | 0.9844 |     0.000331 |     0.005342 | 0.9844 |

## Analysis
- Both implementations achieve identical accuracy, confirming correctness.  
- KD-Tree has slightly higher build time due to tree construction.  
- scikit-learn is faster in practice because it is implemented in optimized C/C++.  
- KD-Tree provides algorithmic scalability and becomes beneficial when implemented in optimized low-level code for very large datasets.

## Conclusion
This project demonstrates how data structures improve the scalability of ML algorithms, and how real-world performance depends on both algorithmic design and systems-level optimization.
