# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Initialize Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Step 6: Train the classifier
gnb.fit(X_train, y_train)

# Step 7: Predict on test data
y_pred = gnb.predict(X_test)

# Step 8: Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)


# Step 9: Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
