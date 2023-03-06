# Cluster Based Knn Imputation
see all the results and experiments [here](https://github.com/YD5463/TabularDataProject/blob/master/main.ipynb)

Note: All the code is in the main.ipynb notebook
In addition you can read on the project in pdf attached to this repository.

## Overview on the project

Filling null values in data observation is one of the major steps in data science pipeline.
This is even more crucial for cases where there have small datasets. In our project we focus on the KNN imputation method, but instead to determining the same k which we got from the user for every sample, we use a clustering method to adjust for each sample with a nan value the cluster from which he is likely to originate and evaluate the clustering density on the cluster. Thus, as the density increases, the k will increase. we run our method on multiple datasets and compare it with classic KNN imputation, we found that our method improve the results in cases where the percentage of null values is high and there have high quality of clustering.

In order to gauge the quality of the method we use, we tested it on several different dataset:

|Name|No.Columns|No. Samples|Explanation|
| - | - | - | - |

|[Titanic](https://www.kaggle.com/competitions/titanic)|8|712|passengers details and survival status|
| - | - | - | - |
|[Mobile](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)|21|2000|mobile phone specifications and prices|
|[Mnist](https://www.kaggle.com/c/digit-recognizer)|64|1797|handwritten digits gray scale images|

