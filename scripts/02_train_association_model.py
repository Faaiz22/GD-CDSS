import sys, os
from copy import deepcopy
import json
import random

# Ensure repo root and src are on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import yaml

from models.dual_branch_net import DualBranchNet

# Paths
DATASET_PATH = os.path.join(ROOT, "artifacts", "association_dataset.pt")
CONFIG_PATH = os.path.join(ROOT, "config", "config.yaml")
MODEL_OUT = os.path.join(ROOT, "artifacts", "model.pt")
METRICS_OUT = os.path.join(ROOT, "artifacts", "association_metrics.json")

# Load config and cast numeric types explicitly
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

train_cfg = cfg.get("models", {}).get("association_model", {})
drug_dim = int(cfg["pipeline"]["drug_vector_dim"])
prot_dim = int(cfg["pipeline"]["protein_vector_dim"])
hidden = int(train_cfg.get("hidden_dim", 256))
dropout = float(train_cfg.get("dropout", 0.25))
lr = float(train_cfg.get("lr", 1e-3))
batch_size = int(train_cfg.get("batch_size", 64))
epochs = int(train_cfg.get("num_epochs", 40))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load dataset
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"association dataset not found at {DATASET_PATH}. Run build artifacts first.")

ds = torch.load(DATASET_PATH)
X_drug = ds["drug"]
X_prot = ds["prot"]
Y = ds["label"]

N = X_drug.shape[0]
if N == 0:
    raise RuntimeError("Association dataset is empty.")

full_ds = TensorDataset(X_drug, X_prot, Y)

# 80/10/10 split (robust)
n_train = int(0.8 * N)
n_val = int(0.1 * N)
n_test = N - n_train - n_val
if n_test <= 0:
    n_test = max(1, N - n_train - n_val)
splits = [n_train, n_val, n_test]
train_ds, val_ds, test_ds = random_split(full_ds, splits, generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Model, optimizer, loss
model = DualBranchNet(drug_dim=drug_dim, prot_dim=prot_dim, hidden_dim=hidden, dropout=dropout).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

best_val_loss = float("inf")
best_state = None
history = {"train_loss": [], "val_loss": []}

for epoch in range(1, epochs + 1):
    model.train()
    running = 0.0
    for d, p, y in train_loader:
        d, p, y = d.to(DEVICE), p.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(d, p)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * d.size(0)
    train_loss = running / len(train_loader.dataset)
    history["train_loss"].append(train_loss)

    # validation
    model.eval()
    vrun = 0.0
    with torch.no_grad():
        for vd, vp, vy in val_loader:
            vd, vp, vy = vd.to(DEVICE), vp.to(DEVICE), vy.to(DEVICE)
            vpreds = model(vd, vp)
            l = criterion(vpreds, vy)
            vrun += l.item() * vd.size(0)
    val_loss = vrun / len(val_loader.dataset)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.6f} — Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = deepcopy(model.state_dict())
        torch.save({"state_dict": best_state, "epoch": epoch, "val_loss": best_val_loss}, MODEL_OUT)
        print(f"  -> New best model saved (val_loss={best_val_loss:.6f})")

# Ensure best_state exists
if best_state is None:
    best_state = deepcopy(model.state_dict())
    torch.save({"state_dict": best_state, "epoch": epochs, "val_loss": best_val_loss}, MODEL_OUT)

# Evaluate on test set
model.load_state_dict(best_state)
model.to(DEVICE)
model.eval()

y_true = []
y_scores = []
with torch.no_grad():
    for td, tp, ty in test_loader:
        td, tp = td.to(DEVICE), tp.to(DEVICE)
        preds = model(td, tp).cpu().numpy().ravel()
        y_scores.extend(preds.tolist())
        y_true.extend(ty.cpu().numpy().ravel().tolist())

# Compute AUC/AUPRC if possible
auc = None
auprc = None
try:
    if len(set(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_scores))
        auprc = float(average_precision_score(y_true, y_scores))
except Exception:
    auc = None
    auprc = None

metrics = {
    "train_config": {
        "batch_size": batch_size,
        "lr": lr,
        "num_epochs": epochs,
        "hidden_dim": hidden,
        "dropout": dropout
    },
    "history": history,
    "best_val_loss": best_val_loss,
    "test": {
        "n_test": len(y_true),
        "roc_auc": auc,
        "auprc": auprc
    }
}

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

print("Training finished.")
print(f"Saved model -> {MODEL_OUT}")
print(f"Saved metrics -> {METRICS_OUT}")
