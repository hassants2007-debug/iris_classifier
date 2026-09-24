scikit-learn>=1.3
matplotlib>=3.8
seaborn>=0.13
jupyter>=1.0
pytest>=8.0
joblib>=1.3
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy:.2f}")
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
xticklabels=iris.target_names,
yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()
# Save model
joblib.dump(model, "outputs/model.joblib")
print("Model and confusion matrix saved in outputs/")
