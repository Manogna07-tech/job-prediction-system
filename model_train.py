import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib


# Load dataset
data = pd.read_csv("job_dataset.csv")

# Create label encoders
degree_encoder = LabelEncoder()
spec_encoder = LabelEncoder()
job_encoder = LabelEncoder()

# Convert text columns into numbers
data["Degree"] = degree_encoder.fit_transform(data["Degree"])
data["Specialization"] = spec_encoder.fit_transform(data["Specialization"])
data["JobRole"] = job_encoder.fit_transform(data["JobRole"])

# Input features
X = data[["Degree", "Specialization", "CGPA"]]

# Output label
y = data["JobRole"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "job_model.pkl")

# Save encoders
joblib.dump(degree_encoder, "degree_encoder.pkl")
joblib.dump(spec_encoder, "spec_encoder.pkl")
joblib.dump(job_encoder, "job_encoder.pkl")

print("Model training completed and saved successfully!")