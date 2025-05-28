# Step 1: Define models suitable for a lightweight IDS setup
# These are models optimized for edge performance while balancing accuracy

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Define lightweight models with fast inference and lower memory requirements
lightweight_models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
}

# Step 2: Train and evaluate models
lightweight_results = []
for name, model in lightweight_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    acc = round(np.mean(y_pred == y_test), 4)
    roc = round(roc_auc_score(y_test, y_score), 4)
    prec, rec, _ = precision_recall_curve(y_test, y_score)
    pr_auc_val = round(auc(rec, prec), 4)

    lightweight_results.append({
        "Model": name,
        "Accuracy": acc,
        "ROC AUC": roc,
        "PR AUC": pr_auc_val
    })

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Lightweight IDS Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# PR Curve
for name, model in lightweight_models.items():
    y_score = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.plot(recall, precision, label=name)

plt.title("Precision-Recall Curve - Lightweight IDS Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()

# Bar chart comparison
df_lightweight_results = pd.DataFrame(lightweight_results).sort_values(by="Accuracy", ascending=False)
df_lightweight_results.plot(kind='bar', x='Model', y=['Accuracy', 'ROC AUC', 'PR AUC'], figsize=(10, 6))
plt.title("Lightweight IDS Model Performance Comparison")
plt.ylabel("Score")
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display the performance table
pd.DataFrame(report).transpose()
df_lightweight_results
