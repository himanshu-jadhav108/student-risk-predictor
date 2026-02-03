import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------- LOAD DATA ----------------
data = pd.read_csv("student_data.csv")

print("📄 Dataset Columns:")
print(list(data.columns))

# ---------------- SAFETY CHECK ----------------
required_cols = [
    'attendance',
    'internal_marks',
    'study_hours',
    'assignments_completed',
    'previous_failures',
    'risk'
]

missing = [col for col in required_cols if col not in data.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# ---------------- FEATURES & TARGET ----------------
X = data[
    [
        'attendance',
        'internal_marks',
        'study_hours',
        'assignments_completed',
        'previous_failures'
    ]
]

y = data['risk']

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # IMPORTANT for imbalanced data
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---------------- TRAIN MODEL ----------------
start_time = time.time()

model = LogisticRegression(
    max_iter=1000,
    solver='liblinear'
)

model.fit(X_train, y_train)

end_time = time.time()

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Training Complete")
print(f"⏱ Training Time: {end_time - start_time:.4f} seconds")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------- FEATURE IMPORTANCE ----------------
print("\n🔎 Feature Importance (Logistic Coefficients):")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature:25s} : {coef:.4f}")

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "student_risk_model.pkl")
print("\n💾 Model saved as student_risk_model.pkl")
