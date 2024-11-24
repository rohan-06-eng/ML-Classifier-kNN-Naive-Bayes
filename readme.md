# Understanding K-Nearest Neighbors (KNN) and Naive Bayes

## Introduction

Machine learning algorithms can be broadly categorized into supervised and unsupervised learning techniques. Among supervised learning algorithms, **K-Nearest Neighbors (KNN)** and **Naive Bayes** are two foundational algorithms used for classification tasks. 

This document provides an overview of these algorithms, demonstrates their working principles, and illustrates practical examples using synthetic and real-world datasets. Specifically, we will explore:
1. **K-Nearest Neighbors (KNN)** using a randomly generated synthetic dataset.
2. **Naive Bayes** using the famous **Iris Dataset**.

## K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** is a simple and intuitive classification algorithm that works by classifying a data point based on the majority label of its 'K' nearest neighbors in the feature space. It is a non-parametric and lazy learning algorithm, meaning it doesn't assume any prior distribution and does not require training before making predictions.

### Working Principle of KNN
1. **Choose the number of neighbors (K)**: The value of K determines how many neighbors the algorithm will look at when making a prediction.
2. **Distance Metric**: KNN uses distance metrics like Euclidean distance to calculate how close or far the data points are from each other.
3. **Class Prediction**: For a new data point, the algorithm finds the K closest points in the training dataset, and the majority class label among these K neighbors is assigned to the new data point.

### Steps for KNN with a Randomly Generated Synthetic Dataset:
1. **Generate Synthetic Data**: We create a synthetic dataset with multiple features and labels. This dataset could contain features like height, weight, or any other arbitrary parameters, and the target variable could be a class such as 'A' or 'B'.
2. **Train the KNN Model**: For each data point in the synthetic dataset, we calculate the distance to all other points. We then select the K nearest neighbors based on the smallest distance values.
3. **Classification**: Once the K nearest neighbors are identified, we classify the new data point based on the majority vote from its K nearest neighbors.
4. **Evaluate the Model**: We evaluate the model’s performance by checking how many predictions are correctly classified on a test set.

## Naive Bayes Classification

**Naive Bayes** is a probabilistic machine learning algorithm based on **Bayes' Theorem**, which is used for classification tasks. The algorithm assumes that the features used for classification are independent of each other, hence the term "naive." Despite the strong assumption of independence, Naive Bayes often performs very well, particularly in text classification and high-dimensional data.

### Working Principle of Naive Bayes

Naive Bayes applies **Bayes' Theorem** to calculate the posterior probability of each class given the input features, and the class with the highest probability is assigned as the predicted class. The theorem is expressed as:

\[
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
\]

Where:
- \(P(C|X)\) is the posterior probability of class \(C\) given the features \(X\).
- \(P(X|C)\) is the likelihood (probability of observing the features \(X\) given class \(C\)).
- \(P(C)\) is the prior probability of the class \(C\).
- \(P(X)\) is the evidence (probability of the features \(X\)).

### Types of Naive Bayes Models

1. **Gaussian Naive Bayes**: Assumes that the features follow a Gaussian (normal) distribution.
2. **Multinomial Naive Bayes**: Used when features represent counts or frequencies (e.g., text classification).
3. **Bernoulli Naive Bayes**: Suitable for binary/boolean features.

### Steps for Naive Bayes with the Iris Dataset

1. **Load the Iris Dataset**: The Iris dataset is a classic dataset in machine learning, containing 150 samples of iris flowers with four features (sepal length, sepal width, petal length, petal width) and three class labels (Setosa, Versicolor, and Virginica).
2. **Preprocess Data**: The dataset is typically already clean, but it may require encoding of categorical features or scaling of numerical features in some cases.
3. **Train the Naive Bayes Model**: The Naive Bayes classifier calculates the prior probabilities and the likelihood of each feature for each class.
4. **Make Predictions**: For a given test instance, Naive Bayes calculates the posterior probability for each class and assigns the class with the highest probability.
5. **Evaluate the Model**: Model performance can be evaluated using metrics such as accuracy, precision, recall, and the confusion matrix.