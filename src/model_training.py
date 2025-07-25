import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

def evaluate_model(y_test, y_pred, y_scores, model_name):
    print(f"\n=== {model_name} Evaluation ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR: {auc_pr:.4f}")

    f1 = f1_score(y_test, y_pred)
    print(f"F1-Score: {f1:.4f}")

    # Plot Precision-Recall Curve
    plt.plot(recall, precision, label=f"{model_name} (AUC-PR = {auc_pr:.2f})")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/cleaned_fraud_data.csv')

    # Drop unwanted columns and separate features/target
    X = df.drop(['class', 'device_id'], axis=1)
    y = df['class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Identify feature types
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # SMOTE + Preprocessing Pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42))
    ])

    # Fit-transform training data
    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

    # Only transform test data (no SMOTE)
    X_test_transformed = preprocessor.transform(X_test)

    # === Logistic Regression ===
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_resampled, y_train_resampled)
    y_pred_logreg = logreg.predict(X_test_transformed)
    y_scores_logreg = logreg.predict_proba(X_test_transformed)[:, 1]

    # === Random Forest ===
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf.predict(X_test_transformed)
    y_scores_rf = rf.predict_proba(X_test_transformed)[:, 1]

    # Save model and preprocessor
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/random_forest_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("Models saved to 'models/' directory.")

    # === Evaluate models ===
    plt.figure(figsize=(8, 6))
    evaluate_model(y_test, y_pred_logreg, y_scores_logreg, "Logistic Regression")
    evaluate_model(y_test, y_pred_rf, y_scores_rf, "Random Forest")

    # Show PR curve
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
