import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# 1. Load data
# -----------------------------
FILE_PATH = "Telepass_assignment.xlsx"
SHEET_NAME = "Sheet1"

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

print("Data shape:", df.shape)
print("Columns:", list(df.columns))


# -----------------------------
# 2. Define target and features
# -----------------------------
target = "y_issued"

feature_cols = [
    "driving_type",
    "car_brand",
    "car_model_group",
    "county_group",
    "broker_group",
    "operating_system",
    "quote_month",
    "price_sale",
    "discount_percent",
]

df = df[feature_cols + [target]].copy()

# Drop rows with missing target
df = df.dropna(subset=[target]).copy()

# Ensure target is numeric 0/1
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(int)

# -----------------------------
# 3. Feature groups
# -----------------------------
categorical_features = [
    "driving_type",
    "car_brand",
    "car_model_group",
    "county_group",
    "broker_group",
    "operating_system",
    "quote_month",
]

numeric_features = [
    "price_sale",
    "discount_percent",
]

# Force categorical columns to consistent string type
for col in categorical_features:
    df[col] = df[col].fillna("Missing").astype(str)

# Force numeric columns to proper numeric type
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Optional quick debug check
print("\nColumn type check:")
for col in feature_cols:
    print(col, df[col].map(type).value_counts().to_dict())

X = df[feature_cols]
y = df[target]

print("\nTarget distribution:")
print(y.value_counts(normalize=True))


# -----------------------------
# 4. Preprocessing
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# -----------------------------
# 5. Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 6. Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    ),
}


# -----------------------------
# 7. Train and evaluate
# -----------------------------
results = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Fit model
    pipe.fit(X_train, y_train)

    # Predict probabilities
    train_pred_prob = pipe.predict_proba(X_train)[:, 1]
    test_pred_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred_class = (test_pred_prob >= 0.5).astype(int)

    # Main metrics
    train_logloss = log_loss(y_train, train_pred_prob)
    test_logloss = log_loss(y_test, test_pred_prob)

    train_auc = roc_auc_score(y_train, train_pred_prob)
    test_auc = roc_auc_score(y_test, test_pred_prob)

    test_accuracy = accuracy_score(y_test, test_pred_class)

    # Cross-validated log loss
    cv_neg_logloss = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=-1
    )
    cv_logloss_mean = -cv_neg_logloss.mean()
    cv_logloss_std = cv_neg_logloss.std()

    results.append({
        "Model": model_name,
        "Train Log Loss": train_logloss,
        "Test Log Loss": test_logloss,
        "CV Log Loss Mean": cv_logloss_mean,
        "CV Log Loss Std": cv_logloss_std,
        "Train AUC": train_auc,
        "Test AUC": test_auc,
        "Test Accuracy": test_accuracy,
    })

results_df = pd.DataFrame(results).sort_values("Test Log Loss").reset_index(drop=True)

print("\nModel comparison:")
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]["Model"]
print(f"\nBest model by Test Log Loss: {best_model}")