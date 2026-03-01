import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the training data
df = pd.read_csv('data/train.csv')

X = df.iloc[:, :-1]
y = df['target']

# Train the model
# Create model with reproducible results
model = RandomForestClassifier(random_state=42)
model.fit(X, y) # [cite: 308]

# Save the trained model file
joblib.dump(model, 'model.joblib')

print("Training complete: model saved as model.joblib")
