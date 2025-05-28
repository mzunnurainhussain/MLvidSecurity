# STEP 1: Install Required Packages
!pip install gdown scikit-learn pandas matplotlib seaborn --quiet

# STEP 2: Download Dataset using gdown
import gdown

file_id = "1M1CN1zc3mH9MCyUL4fU25OYgh9fMGZXZ"
url = f"https://drive.google.com/uc?id={file_id}"
output = "inddos24.csv"
gdown.download(url, output, quiet=False)

# STEP 3: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# STEP 4: Load Dataset
df = pd.read_csv("inddos24.csv")

# STEP 5: Preprocess Dataset
# Drop high-cardinality and non-informative columns
df = df.drop(columns=['Timestamp', 'Source IP', 'Destination IP', 'Firmware Version'], errors='ignore')

# Encode all categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Check for missing values
df = df.dropna()

# Optional: Convert to binary classification (normal vs. attack)
if 'Attack Type' in df.columns:
    df['Target'] = df['Attack Type'].apply(lambda x: 0 if x == 0 or x == 'No Attack' else 1)
    y = df['Target']
else:
    y = df['Labels']

X = df.drop(columns=['Labels', 'Attack Type', 'Target'], errors='ignore')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# STEP 6: Train Multiple Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, preds, average='weighted', zero_division=0)
    })

# STEP 7: Display Results
results_df = pd.DataFrame(results)

# Visualize Accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# Visualize F1 Score
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title("Model F1 Score Comparison")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# Display table
results_df




from sklearn.preprocessing import MinMaxScaler

# Separate scalers for different model needs
X_standard = StandardScaler().fit_transform(X)
X_minmax = MinMaxScaler().fit_transform(X)

# Train-test split for both
X_train_std, X_test_std, y_train, y_test = train_test_split(X_standard, y, test_size=0.3, random_state=42, stratify=y)
X_train_mm, X_test_mm, _, _ = train_test_split(X_minmax, y, test_size=0.3, random_state=42, stratify=y)

# Re-defined models with correct input scaling
models_improved = {
    "XGBoost": (X_train_std, X_test_std, XGBClassifier(eval_metric='logloss')),
    "LightGBM": (X_train_std, X_test_std, LGBMClassifier()),
    "Logistic Regression": (X_train_std, X_test_std, LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=500)),
    "Random Forest (Tuned)": (X_train_std, X_test_std, RandomForestClassifier(class_weight='balanced', max_depth=20)),
    "Extra Trees": (X_train_std, X_test_std, ExtraTreesClassifier()),
    "Complement NB": (X_train_mm, X_test_mm, ComplementNB())
}

# Evaluate
results_improved = []
for name, (Xtr, Xte, model) in models_improved.items():
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    results_improved.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, preds, average='weighted', zero_division=0)
    })

# Display results
results_df_final = pd.DataFrame(results_improved)

# Plot results
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df_final)
plt.title("Improved Accuracy Comparison (All Models)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1 Score", data=results_df_final)
plt.title("Improved F1 Score Comparison (All Models)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

results_df_final


# STEP 1: Install necessary packages (only if needed in Colab)
# !pip install lightgbm scikit-learn pandas matplotlib seaborn --quiet

# STEP 2: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from lightgbm import LGBMClassifier

# STEP 3: Load your data (replace with your DataFrame if already loaded)
df = pd.read_csv("inddos24.csv")  # Update path as needed
df = df.drop(columns=['Timestamp', 'Source IP', 'Destination IP', 'Firmware Version'], errors='ignore')
df = df.dropna()

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Define features and target
X = df.drop(columns=['Labels', 'Attack Type'], errors='ignore')
y = df['Labels'] if 'Labels' in df.columns else df['Target']

# STEP 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    StandardScaler().fit_transform(X), y, test_size=0.3, stratify=y, random_state=42
)

# STEP 5: Define Base Models
lgb = LGBMClassifier()
rf = RandomForestClassifier(class_weight='balanced')
et = ExtraTreesClassifier()
estimators = [
    ('lgb', lgb),
    ('rf', rf),
    ('et', et)
]

# STEP 6: Voting Classifier + GridSearchCV
voting_clf = VotingClassifier(estimators=estimators, voting='soft')

param_grid = {
    'lgb__n_estimators': [100],
    'lgb__learning_rate': [0.1, 0.05],
    'rf__max_depth': [10, 20],
    'et__max_depth': [10, 20]
}

grid = GridSearchCV(estimator=voting_clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

grid_preds = grid.best_estimator_.predict(X_test)
grid_result = {
    "Model": "Tuned VotingClassifier (GridSearchCV)",
    "Accuracy": accuracy_score(y_test, grid_preds),
    "Precision": precision_score(y_test, grid_preds, average='weighted', zero_division=0),
    "Recall": recall_score(y_test, grid_preds, average='weighted', zero_division=0),
    "F1 Score": f1_score(y_test, grid_preds, average='weighted', zero_division=0)
}

# STEP 7: Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500, class_weight='balanced'),
    passthrough=True,
    cv=3,
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
stacking_preds = stacking_clf.predict(X_test)

stacking_result = {
    "Model": "StackingClassifier (LogReg Meta)",
    "Accuracy": accuracy_score(y_test, stacking_preds),
    "Precision": precision_score(y_test, stacking_preds, average='weighted', zero_division=0),
    "Recall": recall_score(y_test, stacking_preds, average='weighted', zero_division=0),
    "F1 Score": f1_score(y_test, stacking_preds, average='weighted', zero_division=0)
}

# STEP 8: Combine and Visualize Results
results_df_final = pd.DataFrame([grid_result, stacking_result])

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df_final)
plt.title("Final Accuracy: Voting + Stacking")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1 Score", data=results_df_final)
plt.title("Final F1 Score: Voting + Stacking")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Performance Summary:")
display(results_df_final)

# STEP 9: Confusion Matrix
cm = confusion_matrix(y_test, stacking_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: StackingClassifier")
plt.grid(False)
plt.show()

# STEP 10: Export Predictions and Model
predictions_df = pd.DataFrame({
    "True Label": y_test.values,
    "Predicted Label": stacking_preds
})
predictions_df.to_csv("stacking_predictions.csv", index=False)
joblib.dump(stacking_clf, "stacking_model.joblib")



# STEP 1: Install Required Packages (only run in Google Colab)
!pip install gdown scikit-learn pandas matplotlib seaborn --quiet

# STEP 2: Download Dataset using gdown
import gdown
file_id = "1M1CN1zc3mH9MCyUL4fU25OYgh9fMGZXZ"
url = f"https://drive.google.com/uc?id={file_id}"
output = "inddos24.csv"
gdown.download(url, output, quiet=False)

# STEP 3: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# STEP 4: Load Dataset
df = pd.read_csv("inddos24.csv")
df.drop(columns=['Timestamp', 'Source IP', 'Destination IP', 'Firmware Version'], errors='ignore', inplace=True)
df.dropna(inplace=True)

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Define features and target
X = df.drop(columns=['Labels'], errors='ignore')
y = df['Labels']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# Feature selection using Extra Trees
selector = SelectFromModel(ExtraTreesClassifier(n_estimators=100), threshold='median')
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# Define base models (LightGBM replaced with GradientBoosting)
rf = RandomForestClassifier(n_estimators=300, max_depth=30, class_weight='balanced')
et = ExtraTreesClassifier(n_estimators=300, max_depth=30)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)

# Voting Ensemble (Soft Voting)
voting = VotingClassifier(
    estimators=[('rf', rf), ('et', et), ('gb', gb)],
    voting='soft',
    n_jobs=-1
)

# Train VotingClassifier
voting.fit(X_train_sel, y_train)
voting_preds = voting.predict(X_test_sel)

# Evaluate
results = {
    "Model": "VotingClassifier (RF+ET+GB)",
    "Accuracy": accuracy_score(y_test, voting_preds),
    "Precision": precision_score(y_test, voting_preds, average='weighted', zero_division=0),
    "Recall": recall_score(y_test, voting_preds, average='weighted', zero_division=0),
    "F1 Score": f1_score(y_test, voting_preds, average='weighted', zero_division=0)
}
results_df = pd.DataFrame([results])

# Visualize accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("VotingClassifier Accuracy")
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, voting_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: VotingClassifier")
plt.show()

# Export predictions and model
pd.DataFrame({"True Label": y_test, "Predicted Label": voting_preds}).to_csv("voting_predictions.csv", index=False)
joblib.dump(voting, "voting_model.joblib")



# STEP 1: Install Required Packages (only needed in Colab)
!pip install gdown scikit-learn pandas matplotlib seaborn --quiet

# STEP 2: Download Dataset using gdown
import gdown
file_id = "1M1CN1zc3mH9MCyUL4fU25OYgh9fMGZXZ"
url = f"https://drive.google.com/uc?id={file_id}"
output = "inddos24.csv"
gdown.download(url, output, quiet=False)

# STEP 3: Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# STEP 4: Load Dataset
df = pd.read_csv("inddos24.csv")
df.drop(columns=['Timestamp', 'Source IP', 'Destination IP', 'Firmware Version'], errors='ignore', inplace=True)
df.dropna(inplace=True)

# STEP 5: Encode Categorical Columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=['Labels'], errors='ignore')
y = df['Labels']

# STEP 6: Normalize & Select Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

selector = SelectFromModel(ExtraTreesClassifier(n_estimators=100), threshold='median')
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# STEP 7: Define Base Models
rf = RandomForestClassifier(n_estimators=300, max_depth=30, class_weight='balanced')
et = ExtraTreesClassifier(n_estimators=300, max_depth=30)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)

# STEP 8: Voting Ensemble
voting = VotingClassifier(
    estimators=[('rf', rf), ('et', et), ('gb', gb)],
    voting='soft',
    n_jobs=-1
)

voting.fit(X_train_sel, y_train)
voting_preds = voting.predict(X_test_sel)

# STEP 9: Evaluation
results = {
    "Model": "VotingClassifier (RF+ET+GB)",
    "Accuracy": accuracy_score(y_test, voting_preds),
    "Precision": precision_score(y_test, voting_preds, average='weighted', zero_division=0),
    "Recall": recall_score(y_test, voting_preds, average='weighted', zero_division=0),
    "F1 Score": f1_score(y_test, voting_preds, average='weighted', zero_division=0)
}
results_df = pd.DataFrame([results])

# STEP 10: Visualizations
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Voting Ensemble Accuracy")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title("Voting Ensemble F1 Score")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, voting_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: VotingClassifier")
plt.show()

# STEP 11: Export Predictions and Model
pd.DataFrame({"True Label": y_test, "Predicted Label": voting_preds}).to_csv("voting_predictions.csv", index=False)
joblib.dump(voting, "voting_model.joblib")


# STEP 1: Install Required Packages (only needed in Colab)
!pip install gdown scikit-learn pandas matplotlib seaborn --quiet

# STEP 2: Download Dataset using gdown
import gdown
file_id = "1M1CN1zc3mH9MCyUL4fU25OYgh9fMGZXZ"
url = f"https://drive.google.com/uc?id={file_id}"
output = "inddos24.csv"
gdown.download(url, output, quiet=False)

# STEP 3: Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# STEP 4: Load Dataset
df = pd.read_csv("inddos24.csv")
df.drop(columns=['Timestamp', 'Source IP', 'Destination IP', 'Firmware Version'], errors='ignore', inplace=True)
df.dropna(inplace=True)

# STEP 5: Encode Categorical Columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=['Labels'], errors='ignore')
y = df['Labels']

# STEP 6: Normalize & Select Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

selector = SelectFromModel(ExtraTreesClassifier(n_estimators=100), threshold='median')
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# STEP 7: Define Base Models
rf = RandomForestClassifier(n_estimators=300, max_depth=30, class_weight='balanced')
et = ExtraTreesClassifier(n_estimators=300, max_depth=30)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)

# STEP 8: Voting Ensemble
voting = VotingClassifier(
    estimators=[('rf', rf), ('et', et), ('gb', gb)],
    voting='soft',
    n_jobs=-1
)

voting.fit(X_train_sel, y_train)
voting_preds = voting.predict(X_test_sel)

# STEP 9: Evaluation
results = {
    "Model": "VotingClassifier (RF+ET+GB)",
    "Accuracy": accuracy_score(y_test, voting_preds),
    "Precision": precision_score(y_test, voting_preds, average='weighted', zero_division=0),
    "Recall": recall_score(y_test, voting_preds, average='weighted', zero_division=0),
    "F1 Score": f1_score(y_test, voting_preds, average='weighted', zero_division=0)
}
results_df = pd.DataFrame([results])

# STEP 10: Visualizations
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Voting Ensemble Accuracy")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title("Voting Ensemble F1 Score")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, voting_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix: VotingClassifier")
plt.show()

# STEP 11: Export Predictions and Model
pd.DataFrame({"True Label": y_test, "Predicted Label": voting_preds}).to_csv("voting_predictions.csv", index=False)
joblib.dump(voting, "voting_model.joblib")
