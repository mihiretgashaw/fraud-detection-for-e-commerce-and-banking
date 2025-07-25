import pandas as pd
import joblib
import shap
import time

# Load Cleaned data
print("[INFO] Loading data...")
df = pd.read_csv("data/processed/cleaned_fraud_data.csv")
X = df.drop(["class", "device_id"], axis=1)
y = df["class"]

# Load preprocessor and model
print("[INFO] Loading preprocessor and model...")
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/random_forest_model.pkl")

# Sample a smaller subset for explainability
print("[INFO] Sampling data...")
X_sample = X.sample(n=200, random_state=42)
X_sample_transformed = preprocessor.transform(X_sample)

# Convert to dense array if needed
if hasattr(X_sample_transformed, "toarray"):
    X_sample_transformed = X_sample_transformed.toarray()

# Get feature names and convert to DataFrame
feature_names = preprocessor.get_feature_names_out()
X_sample_df = pd.DataFrame(X_sample_transformed, columns=feature_names)

# Run SHAP
print("[INFO] Running SHAP...")
start_time = time.time()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample_df)
print(f"[INFO] SHAP completed in {time.time() - start_time:.2f} seconds.")

# Plot summary
print("[INFO] Plotting SHAP summary...")
shap.summary_plot(shap_values[1], X_sample_df)
