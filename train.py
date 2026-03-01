import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load the training data
df = pd.read_csv('data/train.csv')

X = df.iloc[:, :-1]
y = df['target']

# Train the model
# Increased max_iter to 3000 to ensure convergence
model = LogisticRegression(random_state=42, max_iter=3000)
model.fit(X, y)

# Save the trained model file
joblib.dump(model, 'model.joblib')

print("Training complete: model saved as model.joblib")
