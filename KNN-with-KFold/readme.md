
# K-Nearest Neighbors (KNN) Classification Documentation

## Overview
This code implements the K-Nearest Neighbors (KNN) algorithm to classify iris flower species based on their features. It uses a dataset containing measurements of sepal and petal lengths and widths.

## Dependencies
The following libraries are required to run the code:
- pandas
- numpy
- collections
- math
- sklearn

## Code Structure

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
from collections import Counter
import math
from sklearn.model_selection import train_test_split
```

### 2. Data Loading
The dataset is loaded using pandas. It reads the iris dataset from a CSV file and ensures that each feature is in the correct data type (float).
```python
data = pd.read_csv('Lab_A7_iris.data.csv')
data['sepal_length'] = data['sepal_length'].astype(float)
data['sepal_width'] = data['sepal_width'].astype(float)
data['petal_length'] = data['petal_length'].astype(float)
data['petal_width'] = data['petal_width'].astype(float)
```

### 3. Euclidean Distance Calculation
This function calculates the Euclidean distance between two data points.
```python
def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)
```

### 4. Finding K Nearest Neighbors
This function identifies the K nearest neighbors for a given test instance.
```python
def get_neighbors(train, test_row, k):
    distances = []
    for train_row in train.values:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors
```

### 5. KNN Prediction
This function predicts the class label for a test instance based on its nearest neighbors.
```python
def knn_predict(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_values = [row[-1] for row in neighbors]
    return Counter(output_values).most_common(1)[0][0]
```

### 6. K-Fold Cross-Validation
This function performs K-Fold cross-validation to evaluate the model's performance across different training and testing sets.
```python
def k_fold_cross_validation(data, k_folds=4, k_neighbors=4):
    fold_size = len(data) // k_folds
    folds = [data[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]
    accuracy_list = []

    for i in range(k_folds):
        train_set = pd.concat([folds[j] for j in range(k_folds) if j != i])
        test_set = folds[i]

        correct_predictions = 0
        for test_row in test_set.values:
            prediction = knn_predict(train_set, test_row, k_neighbors)
            if prediction == test_row[-1]:
                correct_predictions += 1

        accuracy = correct_predictions / len(test_set)
        accuracy_list.append(accuracy)

    return accuracy_list
```

### 7. Main Execution
The main block of the code splits the data into training and testing sets, evaluates various values of K using cross-validation, and identifies the best K value based on accuracy.
```python
if __name__ == "__main__":
    train_data, test_data = train_test_split(data, test_size=0.2)

    k_values = range(1, 31)
    accuracy_results = []

    for k in k_values:
        accuracy_list = k_fold_cross_validation(train_data, k_folds=4, k_neighbors=k)
        avg_validation_accuracy = np.mean(accuracy_list)
        validation_std_dev = np.std(accuracy_list)
        accuracy_results.append((k, avg_validation_accuracy, validation_std_dev))

    best_k, best_accuracy, best_std_dev = max(accuracy_results, key=lambda x: x[1])

    correct_predictions = 0
    for test_row in test_data.values:
        prediction = knn_predict(train_data, test_row, best_k)
        if prediction == test_row[-1]:
            correct_predictions += 1

    test_accuracy = correct_predictions / len(test_data)

    print(f"Best K value: {best_k}")
    print(f"Validation Accuracy for Best K: {best_accuracy}")
    print(f"Validation Accuracy Standard Deviation for Best K: {best_std_dev}")
    print(f"Test Accuracy with Best K: {test_accuracy}")
```

## Output
Upon execution, the code outputs the following:
- Best K value
- Validation Accuracy for the best K
- Validation Accuracy Standard Deviation for the best K
- Test Accuracy with the best K

This provides a comprehensive view of the model's performance based on different values of K.

## Conclusion
The K-Nearest Neighbors algorithm is implemented to classify iris species based on sepal and petal measurements. The code uses K-fold cross-validation to identify the optimal number of neighbors (K) for the best classification accuracy.
