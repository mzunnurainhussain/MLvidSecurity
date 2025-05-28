!pip install torch torchvision adversarial-robustness-toolbox scikit-learn pandas matplotlib seaborn

# ✅ Improved PyTorch + ART Implementation for UNSW-NB15 IDS (With Visualizations)

# 1. Install Required Libraries
!pip install -q gdown torch torchvision adversarial-robustness-toolbox scikit-learn pandas matplotlib seaborn

# 2. Imports
import gdown
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

# 3. Download Dataset
gdown.download("https://drive.google.com/uc?id=1--bSM-4Xzsxv_IHdJeJnm01Hb0KXBmBx", "UNSW-NB15_Part1.csv", quiet=False)
gdown.download("https://drive.google.com/uc?id=1--rBqY2Y7hpGOSeHcaodrM7D5oBhBXkA", "UNSW-NB15_Part2.csv", quiet=False)

# 4. Data Preprocessing
df = pd.concat([pd.read_csv("UNSW-NB15_Part1.csv"), pd.read_csv("UNSW-NB15_Part2.csv")])
df.dropna(inplace=True)
df['label'] = LabelEncoder().fit_transform(df['label'])
X = df.drop(columns=['label'])
y = df['label']

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 5. Improved Transformer-CNN Model
class EnhancedTransformerCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EnhancedTransformerCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(2)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

num_classes = int(len(np.unique(y)))
model = EnhancedTransformerCNN(X_train.shape[1], num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 6. Train the Model
model.train()
for epoch in range(10):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

# 7. Adversarial Training with FGSM
classifier_train = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer, input_shape=(X_train.shape[1],), nb_classes=num_classes, clip_values=(float(X_train_tensor.min()), float(X_train_tensor.max())))
fgsm = FastGradientMethod(estimator=classifier_train, eps=0.1)
adv_X_train = fgsm.generate(x=X_train_tensor.numpy())
adv_X_train_tensor = torch.tensor(adv_X_train, dtype=torch.float32)

adv_dataset = TensorDataset(adv_X_train_tensor, y_train_tensor)
adv_loader = DataLoader(adv_dataset, batch_size=64, shuffle=True)

print("\nRetraining with adversarial examples...")
model.train()
for epoch in range(5):
    total_loss = 0
    for xb, yb in adv_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Adv Epoch {epoch+1}: Loss = {total_loss/len(adv_loader):.4f}")

# 8. FGSM and PGD Attacks
classifier = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer, input_shape=(X_train.shape[1],), nb_classes=num_classes, clip_values=(float(X_train_tensor.min()), float(X_train_tensor.max())))
attack_fgsm = FastGradientMethod(estimator=classifier, eps=0.1)
attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=40)

X_test_adv_fgsm = attack_fgsm.generate(X_test_tensor.numpy())
X_test_adv_pgd = attack_pgd.generate(X_test_tensor.numpy())

# 9. Evaluate
model.eval()
def evaluate(model, X, y):
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        print(classification_report(y, preds))
        return preds, probs

print("\nFGSM Attack Results:")
preds_fgsm, probs_fgsm = evaluate(model, X_test_adv_fgsm, y_test)
print("\nPGD Attack Results:")
preds_pgd, probs_pgd = evaluate(model, X_test_adv_pgd, y_test)

# 10. Visualizations

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(y_test, preds_fgsm, "FGSM Confusion Matrix")
plot_confusion_matrix(y_test, preds_pgd, "PGD Confusion Matrix")

# ROC Curve
def plot_roc(y_true, y_probs, title):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# PR Curve
def plot_pr(y_true, y_probs, title):
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    plt.figure()
    plt.plot(recall, precision, lw=2, color='green')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.show()

plot_roc(y_test, probs_fgsm, "ROC Curve - FGSM")
plot_roc(y_test, probs_pgd, "ROC Curve - PGD")
plot_pr(y_test, probs_fgsm, "PR Curve - FGSM")
plot_pr(y_test, probs_pgd, "PR Curve - PGD")

