from src.preprocess import load_iris_dataset, load_data, load_mnist, load_titanic, load_mobile_price_dataset
from src.models import density_based_knn, eval_methods_over_k

if __name__ == '__main__':
    # df, missing_values = load_data()
    df, y_true = load_mobile_price_dataset(nan_percentage=0.2)
    X = df.values
    print(df.shape)
    print(load_mnist(nan_percentage=0.2)[0].shape)
    print(load_titanic(nan_percentage=0.2)[0].shape)
    # eval_methods_over_k(X, y_true)
    # scores = density_based_knn(X, y_true)
    # print(scores)
    # plt.scatter(X[:,0], X[:, 1])
    # plt.show()
