from src.preprocess import load_titanic
from src.models import cluster_data, reduce_dimension, knn_imputer
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__ == '__main__':
    df, missing_values = load_titanic()
    X = df.values
    best_clusters = cluster_data(X)
    all_scores = defaultdict(list)
    k_range = list(range(15, 60))
    for k in k_range:
        scores = knn_imputer(best_clusters, X, missing_values.copy(), k)
        for key, value in scores.items():
            all_scores[key].append(value)
    for method_name, scores in all_scores.items():
        plt.plot(k_range, scores, label=method_name)
    plt.legend()
    plt.title("Method scores over K")
    plt.show()
