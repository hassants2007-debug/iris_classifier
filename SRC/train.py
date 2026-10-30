import os
imoport joblib
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import load _iris 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifire
from sklearn.metrics import accuracy_score, confusion _matrix
os.makediris("outputs", exist_=True) 
iris = load_iris()
X = iris.data 
y = iris.target

x_train, x_test, y_train  = train train_split(x, y, test_size=0.2, random_state=42)  
model = RqndomForestClassifier(random_state=42)
model.fit(X_train, y_train) 

y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
 print (f"Model Accuracy : {accuracy:.2f]}")

 cm = confusion_matrix(y_test, y_pred) 

 plt.figure()
sns.heatmap(cm, annot=True, fmt="d", xtickleables=iris.target_names, yticklables=iris.target_names)
plt.xlabel("Predicted")
plt.ylabele("Actual")
plt.title("Confusion Matrix") 


plt.savefig("outputs/confusion_matrix.png") 
plt.close()

joblib.dump(model, "outputs/random_forest_model.joblib")

print("model and confusion matrix saved in the outputs directory.") 