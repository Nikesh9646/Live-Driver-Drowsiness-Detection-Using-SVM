import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset (replace with actual loading code)
# X = features, y = labels (0=Alert, 1=Drowsy)
X = np.load("features.npy")  # e.g., EAR values
y = np.load("labels.npy")    # 0 for Alert, 1 for Drowsy

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(svm, f)
