# KNN Implementation from Scratch

This project demonstrates how to implement the K-Nearest Neighbors (KNN) algorithm from scratch and compares its performance with the scikit-learn library's implementation using the Iris dataset.

## Introduction
The K-Nearest Neighbors (KNN) algorithm is a simple, yet powerful classification algorithm. This project aims to provide a comprehensive guide to implementing KNN from scratch and comparing it with the well-established scikit-learn library.

## Installation
Ensure you have the necessary libraries installed. The required libraries include:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install them using the following command:
```
pip install -U scikit-learn pandas numpy matplotlib seaborn
```

## Dataset
The Iris dataset, a classic dataset in the machine learning community, is used in this project. It contains 150 samples of iris flowers, each with four features (sepal length, sepal width, petal length, petal width) and a class label indicating the species of the flower.

## Data Exploration and Preprocessing
Before applying the KNN algorithm, the dataset is explored and preprocessed:
- **Dataset Information**: Display basic information about the dataset.
- **Head and Description**: View the first few rows and statistical summaries of the dataset.
- **Class Distribution**: Ensure the dataset is balanced.
- **Missing Data**: Check for any missing values.
- **Duplicates**: Identify and remove duplicate entries.
- **Feature and Label Separation**: Split the dataset into features (X) and labels (Y).
- **Data Splitting**: Split the dataset into training and testing sets (80% training, 20% testing).
- **Normalization**: Normalize the features to ensure fair distance calculations in KNN.

## KNN Implementation
The KNN algorithm is implemented from scratch following these steps:
1. **Calculate Euclidean Distance**: Compute the distance between the test point and all training points.
2. **Find Nearest Neighbors**: Identify the k-nearest neighbors to the test point.
3. **Majority Voting**: Determine the class of the test point based on the majority vote of its nearest neighbors.

## Comparison with Scikit-learn
The custom KNN implementation is compared with the scikit-learn library's implementation:
- **Training and Prediction**: Train the scikit-learn KNN model and make predictions on the test set.
- **Accuracy Comparison**: Evaluate and compare the accuracy of both implementations.

## Visualization
Visualizations are used to understand the dataset and the results:
- **Pair Plot**: Display relationships between features before and after normalization.
- **Correlation Heatmap**: Show correlations between features.

## Results
The results section includes:
- **Predictions**: Output of predictions from both the custom and scikit-learn KNN implementations.
- **Accuracy**: Comparison of accuracy scores between the two implementations.
- **Testing with Different k Values**: Results for different values of k (e.g., 3, 5, 7) to see the impact on accuracy.

