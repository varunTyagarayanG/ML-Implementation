import pandas as pd
import numpy as np
from collections import Counter
import math
from sklearn.model_selection import train_test_split

# I have imported the data using pandas
data = pd.read_csv('Lab_A7_iris.data.csv')

# Emphasizing the data type of each field to avoid feature errors!
data['sepal_length'] = data['sepal_length'].astype(float)
data['sepal_width'] = data['sepal_width'].astype(float)
data['petal_length'] = data['petal_length'].astype(float)
data['petal_width'] = data['petal_width'].astype(float)

# Euclidean distance calculation
def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):  # Exclude the label (species)
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# Get the k nearest neighbors
def get_neighbors(train, test_row, k):
    distances = []
    for train_row in train.values:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    distances.sort(key=lambda x: x[1])  # Sort by distance in ascending order
    neighbors = [distances[i][0] for i in range(k)]  # Get the k nearest training rows
    return neighbors

# KNN prediction
def knn_predict(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_values = [row[-1] for row in neighbors]  # Get the labels of the neighbors
    return Counter(output_values).most_common(1)[0][0]  # Return the most common label

# Step 2: K-fold cross-validation
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
            if prediction == test_row[-1]:  # Check if the prediction is correct
                correct_predictions += 1

        accuracy = correct_predictions / len(test_set)
        accuracy_list.append(accuracy)

    return accuracy_list

if __name__ == "__main__":
    # Step 1: Split data into train and test sets using sklearn's train_test_split
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Initialize lists to store K values and their corresponding accuracies
    k_values = range(1, 31)  # Test K values from 1 to 31
    accuracy_results = []

    # Step 2: Perform K-Fold Cross Validation for each K value
    for k in k_values:
        accuracy_list = k_fold_cross_validation(train_data, k_folds=4, k_neighbors=k)
        avg_validation_accuracy = np.mean(accuracy_list)
        validation_std_dev = np.std(accuracy_list)  # Calculate standard deviation
        accuracy_results.append((k, avg_validation_accuracy, validation_std_dev))

    # Step 3: Find the best K value based on the highest accuracy
    best_k, best_accuracy, best_std_dev = max(accuracy_results, key=lambda x: x[1])

    # Test accuracy with the best K value
    correct_predictions = 0
    for test_row in test_data.values:
        prediction = knn_predict(train_data, test_row, best_k)
        if prediction == test_row[-1]:
            correct_predictions += 1

    test_accuracy = correct_predictions / len(test_data)

    # Reporting the results
    print(f"Best K value: {best_k}")
    print(f"Validation Accuracy for Best K: {best_accuracy}")
    print(f"Validation Accuracy Standard Deviation for Best K: {best_std_dev}")
    print(f"Test Accuracy with Best K: {test_accuracy}")
