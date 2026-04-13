import pandas as pd
from sklearn.model_selection import train_test_split
import math
from collections import Counter

# 1. Load dataset
def load_dataset(filename):
    df = pd.read_csv(filename)
    return df

data = load_dataset('lab2_dataset/q2/Iris.csv')
print(data)
X = data.drop(columns=['Species']).values
y = data['Species'].values

# 2. Split dataset into training and test sets using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 3. Define the kNN algorithm
def euclidean_distance(instance1, instance2):
    return math.sqrt(sum((instance2 - instance1)** 2))

def get_neighbors(X_train, test_instance, k):
    distances = []
    neighbors = []

    for i, point in enumerate(X_train):
        distanceVal = euclidean_distance(point, test_instance)
        distances.append((distanceVal, i, point))

    neighbors = [item for item in sorted(distances)[:k]]
    return neighbors

def get_response(neighbors, y_train):
    prediction = None

    class_labels = [y_train[neighbor[1]] for neighbor in neighbors]
    prediction = Counter(class_labels).most_common(1)[0][0]

    return prediction

# 4. Use the kNN algorithm to predict the class labels of the test set
k = 3
predictions = []
for current_instance in X_test:
    neighbors = get_neighbors(X_train, current_instance, k)
    prediction = get_response(neighbors, y_train)
    predictions.append(prediction)

# 5. Calculate the accuracy of the predictions
correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test, predictions)])
accuracy = (correct / len(y_test)) * 100.0
print(f"Accuracy: {accuracy:.2f}%")