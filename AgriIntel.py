# Re-run the pipeline without using ace_tools
import pandas as pd
import gdown
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import StackingClassifier

# Download dataset
url = "https://drive.google.com/uc?id=1sxDecKc8sN30vOEL6aQ2xYwjEqNW5c8d"
gdown.download(url, "combined.csv", quiet=False)
df = pd.read_csv("combined.csv")

# Filter and clean fertilizer prediction data
df_fert = df.dropna(subset=['fertilizer']).copy()
df_fert.drop(columns=df_fert.columns[df_fert.isna().all()], inplace=True)

# Map fertilizer categories
fert_map = {
    'Urea': 'Nitrogen', 'DAP': 'Phosphorus', '10-26-26': 'Phosphorus',
    'Super phosphate': 'Phosphorus', 'Potash': 'Potassium',
    'Complex': 'Balanced', '14-35-14': 'Balanced', '20-20': 'Balanced',
    '28-28': 'Balanced', '17-17-17': 'Balanced'
}
df_fert['fertilizer'] = df_fert['fertilizer'].map(fert_map).fillna("Other")

# Encode categorical features
for col in ['Soil Type', 'label', 'fertilizer']:
    if col in df_fert.columns:
        df_fert[col] = LabelEncoder().fit_transform(df_fert[col].astype(str))

# Feature engineering
df_fert['N_K_ratio'] = df_fert['N'] / (df_fert['K'] + 1)
df_fert['P_K_ratio'] = df_fert['P'] / (df_fert['K'] + 1)

# Categorical bucketing
df_fert['temp_cat'] = pd.cut(df_fert.get('temperature', pd.Series([25]*len(df_fert))), bins=[0, 15, 25, 35, 100], labels=[0,1,2,3])
df_fert['ph_cat'] = pd.cut(df_fert.get('ph', pd.Series([7]*len(df_fert))), bins=[0, 5.5, 6.5, 7.5, 14], labels=[0,1,2,3])
df_fert['hum_cat'] = pd.cut(df_fert.get('humidity', pd.Series([50]*len(df_fert))), bins=[0, 40, 60, 80, 100], labels=[0,1,2,3])

# Feature selection
possible_features = ['N','P','K','temperature','humidity','Moisture','Soil Type','label',
                     'N_K_ratio','P_K_ratio','temp_cat','ph_cat','hum_cat']
features = [col for col in possible_features if col in df_fert.columns]
df_fert.fillna(0, inplace=True)
X = df_fert[features]
y = df_fert['fertilizer']

# Scale and balance the data
X_scaled = StandardScaler().fit_transform(X)
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create base models
cb = CatBoostClassifier(verbose=0, learning_rate=0.05, iterations=200, depth=10)
lgb = LGBMClassifier(learning_rate=0.05, n_estimators=200, max_depth=10, num_leaves=40, subsample=0.8)
xgb = XGBClassifier(learning_rate=0.05, n_estimators=200, max_depth=10, use_label_encoder=False, eval_metric='mlogloss')

# Stacking model
stacked_model = StackingClassifier(
    estimators=[('catboost', cb), ('lightgbm', lgb), ('xgboost', xgb)],
    final_estimator=CatBoostClassifier(verbose=0, learning_rate=0.03, iterations=150)
)
stacked_model.fit(X_train, y_train)
preds = stacked_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, preds)
report = classification_report(y_test, preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot classification report
fig, ax = plt.subplots(figsize=(10, 6))
report_df.iloc[:-3, :-1].plot(kind='bar', ax=ax)
plt.title("Precision, Recall, and F1-Score by Class")
plt.xlabel("Class")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(stacked_model, X_test, y_test, ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix for Fertilizer Prediction")
plt.tight_layout()
plt.show()

report_df, accuracy


pip install catboost lightgbm xgboost imbalanced-learn scikit-learn pandas matplotlib


# STEP 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier

# STEP 2: Load Dataset Manually (upload your file here or read local file)
df = pd.read_csv("combined.csv")

# STEP 3: Clean and Preprocess Data
df_fert = df.dropna(subset=['fertilizer']).copy()
df_fert.drop(columns=df_fert.columns[df_fert.isna().all()], inplace=True)

fert_map = {
    'Urea': 'Nitrogen', 'DAP': 'Phosphorus', '10-26-26': 'Phosphorus',
    'Super phosphate': 'Phosphorus', 'Potash': 'Potassium',
    'Complex': 'Balanced', '14-35-14': 'Balanced', '20-20': 'Balanced',
    '28-28': 'Balanced', '17-17-17': 'Balanced'
}
df_fert['fertilizer'] = df_fert['fertilizer'].map(fert_map).fillna("Other")

for col in ['Soil Type', 'label', 'fertilizer']:
    if col in df_fert.columns:
        df_fert[col] = LabelEncoder().fit_transform(df_fert[col].astype(str))

df_fert['N_K_ratio'] = df_fert['N'] / (df_fert['K'] + 1)
df_fert['P_K_ratio'] = df_fert['P'] / (df_fert['K'] + 1)
df_fert['temp_cat'] = pd.cut(df_fert.get('temperature', pd.Series([25]*len(df_fert))), bins=[0, 15, 25, 35, 100], labels=[0,1,2,3])
df_fert['ph_cat'] = pd.cut(df_fert.get('ph', pd.Series([7]*len(df_fert))), bins=[0, 5.5, 6.5, 7.5, 14], labels=[0,1,2,3])
df_fert['hum_cat'] = pd.cut(df_fert.get('humidity', pd.Series([50]*len(df_fert))), bins=[0, 40, 60, 80, 100], labels=[0,1,2,3])

# STEP 4: Feature Engineering
possible_features = ['N','P','K','temperature','humidity','Moisture','Soil Type','label',
                     'N_K_ratio','P_K_ratio','temp_cat','ph_cat','hum_cat']
features = [col for col in possible_features if col in df_fert.columns]
df_fert.fillna(0, inplace=True)
X = df_fert[features]
y = df_fert['fertilizer']

# STEP 5: Scale + Balance + Split
X_scaled = StandardScaler().fit_transform(X)
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# STEP 6: Define Hybrid Stacking Model
cb = CatBoostClassifier(verbose=0, learning_rate=0.05, iterations=200, depth=10)
lgb = LGBMClassifier(learning_rate=0.05, n_estimators=200, max_depth=10, num_leaves=40, subsample=0.8)
xgb = XGBClassifier(learning_rate=0.05, n_estimators=200, max_depth=10, use_label_encoder=False, eval_metric='mlogloss')

stacked_model = StackingClassifier(
    estimators=[('catboost', cb), ('lightgbm', lgb), ('xgboost', xgb)],
    final_estimator=CatBoostClassifier(verbose=0, learning_rate=0.03, iterations=150)
)
stacked_model.fit(X_train, y_train)

# STEP 7: Evaluate Model
preds = stacked_model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"\n✅ Final Hybrid Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, preds))

# Optional: Show Confusion Matrix
plt.figure(figsize=(10,6))
ConfusionMatrixDisplay.from_estimator(stacked_model, X_test, y_test, cmap="Blues")
plt.title("Fertilizer Classification Confusion Matrix")
plt.show()
