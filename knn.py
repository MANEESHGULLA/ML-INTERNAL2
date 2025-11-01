from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
x_train_sclaed=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

knn=KNeighborsClassifier(n_neighbors=3,metric='manhattan')
knn.fit(x_train,y_train)

unique,counts = np.unique(y_test,return_counts=True)

print(f"Count number of flowers in each category of test set")
for cls,count in zip(unique,counts):
  print(f"{iris.target_names[cls]}:{count}")
print()

y_pred=knn.predict(x_test)

print("\naccuracy score:",accuracy_score(y_test,y_pred))
print("\nconfusion matrix:\n",confusion_matrix(y_test,y_pred))
print()
cm=confusion_matrix(y_test,y_pred)

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=iris.target_names)


disp.plot(cmap="Blues")
plt.title("Confusion Matrix - KNN")
plt.show()

new_sample = np.array([[5.5, 3.2, 1.5, 0.2]])

scaled_sample=scaler.transform(new_sample)

predicted_class=knn.predict(scaled_sample)[0]
print()
print("New Sample:", scaled_sample)
print(f"{iris.target_names[predicted_class]}:{predicted_class}")
