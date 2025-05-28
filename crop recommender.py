import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and clean data
df = pd.read_csv('Fertilizer Prediction.csv')
df = df[~df['Fertilizer Name'].astype(str).str.contains('2026-10-26')]

# 2. Encode categorical features
for col in ['Soil Type', 'Crop Type']:
    df[col] = LabelEncoder().fit_transform(df[col])

# 3. Split X / y and label-encode the target
X = df.drop('Fertilizer Name', axis=1)
y_raw = df['Fertilizer Name']
le_fertilizer = LabelEncoder()
y = le_fertilizer.fit_transform(y_raw)

# 4. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data  = lgb.Dataset(X_test,  label=y_test, reference=train_data)

# 6. Set parameters
params = {
    'objective':     'multiclass',
    'num_class':     len(le_fertilizer.classes_),
    'metric':        'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves':    31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'verbose':         -1,
    'seed':            42
}

# 7. Train with callback-based early stopping & log every 10 rounds
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=10)
    ]
)

# 8. Predict and decode labels
y_pred_prob   = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_class  = np.argmax(y_pred_prob, axis=1)
y_test_decoded     = le_fertilizer.inverse_transform(y_test)
y_pred_decoded     = le_fertilizer.inverse_transform(y_pred_class)

# 9. Evaluation
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# 10. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_matrix(y_test_decoded, y_pred_decoded),
    annot=True, fmt='d',
    xticklabels=le_fertilizer.classes_,
    yticklabels=le_fertilizer.classes_
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 11. Feature Importance
lgb.plot_importance(model, max_num_features=10, figsize=(10, 6))
plt.title('Feature Importance')
plt.show()
