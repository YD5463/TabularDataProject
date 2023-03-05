# Cluster Based Knn Imputation
see all the results and experiments [here](https://github.com/YD5463/TabularDataProject/blob/master/main.ipynb)

Filling null values in data observation is one of the major steps in data science pipeline. This is even more crucial for cases where there have small datasets. In our research we focus on the KNN imputation method, but instead to determine for each sample the same k, we use clustering method to adapt for each of the samples with nan value the cluster which he most probably from there and run the KNN on this and only this neighbours. we run our method on multiple datasets and we found that our method improve the results in cases where the percentage of null values is high.

In order to gauge the quality of the method we use, we tested it on several different dataset:

|Name|No.Columns|No. Samples|Explanation|
| - | - | - | - |

|[Titanic](https://www.kaggle.com/competitions/titanic)|8|712|passengers details and survival status|
| - | - | - | - |
|[Mobile](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)|21|2000|mobile phone specifications and prices|
|[Mnist](https://www.kaggle.com/c/digit-recognizer)|64|1797|handwritten digits gray scale images|

Table 1: Datasets details

## Solution overview

The k-nearest neighbors (KNN) imputation method can be improved by incorporating clustering techniques. Clustering is a method of grouping similar observations together based on their characteristics. By grouping similar observations together, clustering can help to identify regions of high density in the feature space, which can be used to improve the KNN impu- tation method.

In order to estimate the missing value, instead of looking at a constant k that we received from the user, we will use the compression of each cluster, that is, the more compressed a cluster is, the greater the number of neigh- bors, and vice versa

There are two ways to evaluate each imputation method:

1. make nan of features that we already know his actual value and calac- ulate the mean squared error between the real value and the imputed value
1. mpute real missing value and run a baseline classification model on im- puted values and see who gives the best results on the end-problem(accuracy, precision, recall etc.)
