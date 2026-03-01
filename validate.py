import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the test data and the trained model
df = pd.read_csv('data/test.csv')
X_test = df.iloc[:, :-1]
y_test = df['target']

model = joblib.load('model.joblib')

# 2. Make predictions
preds = model.predict(X_test)

# 3. Calculate and save metrics
acc = accuracy_score(y_test, preds)

with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)

# 4. Generate and save the confusion matrix plot
cm = confusion_matrix(y_test, preds, labels=model.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

print(f"Validation complete: Accuracy = {acc:.4f}. Outputs saved to metrics.json and confusion_matrix.png.")
