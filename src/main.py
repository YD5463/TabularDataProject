from src.preprocess import load_titanic, load_data
from src.models import density_based_knn, eval_methods_over_k

if __name__ == '__main__':
    # df, missing_values = load_data()
    df, y_true = load_titanic(nan_percentage=0.3)
    X = df.values
    eval_methods_over_k(X, y_true)
    scores = density_based_knn(X, y_true)
    print(scores)
    # plt.scatter(X[:,0], X[:, 1])
    # plt.show()
