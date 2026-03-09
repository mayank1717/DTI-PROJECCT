import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset (using the project dataset location)
data = pd.read_csv(r"data\\nanoparticle_catalytic_dataset_5000.csv")

# Remove non-numeric/text columns (your dataset has 'composition' instead of 'Nanoparticle')
for col in ["Nanoparticle", "composition", "particle_id"]:
    if col in data.columns:
        data = data.drop(col, axis=1)

# Target (your dataset uses 'catalytic_activity')
target_col = "Catalytic_Activity" if "Catalytic_Activity" in data.columns else "catalytic_activity"
if target_col not in data.columns:
    raise ValueError(f"Target column not found. Available columns: {list(data.columns)}")

# Features / Target
X = data.drop(target_col, axis=1)
y = data[target_col]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model (matches your Streamlit loader)
joblib.dump(model, "catalyst_model.pkl")

print("Model trained successfully!")

