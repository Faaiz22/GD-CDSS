import sys, os
import json
import random
from copy import deepcopy

# --------------------------------------------------------------
# Ensure repo root + src/ are on sys.path
# --------------------------------------------------------------
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
import yaml
from tqdm import tqdm

from models.generative_cvae import GenerativeCVAE, vae_loss_function

# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
ART = os.path.join(ROOT, "artifacts")
POS_PATH = os.path.join(ART, "positive_feature_pairs.pt")
CONFIG_PATH = os.path.join(ROOT, "config", "config.yaml")
MODEL_OUT = os.path.join(ART, "cvae_model.pt")
METRICS_OUT = os.path.join(ART, "cvae_metrics.json")

# --------------------------------------------------------------
# Load config (explicit casting)
# --------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

cvae_cfg = cfg["models"]["cvae"]
input_dim = int(cfg["pipeline"]["drug_vector_dim"])
latent_dim = int(cvae_cfg["latent_dim"])
hidden_dim = int(cvae_cfg["hidden_dim"])
lr = float(cvae_cfg["lr"])
batch_size = int(cvae_cfg["batch_size"])
epochs = int(cvae_cfg["num_epochs"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------------------------------------------------
# Load positive samples
# --------------------------------------------------------------
if not os.path.exists(POS_PATH):
    raise FileNotFoundError(f"positive_feature_pairs.pt not found at {POS_PATH}. Run 01_build_artifacts.py first.")

pos = torch.load(POS_PATH)
if pos.ndim == 1:
    pos = pos.unsqueeze(1)

N = pos.shape[0]
if N < 3:
    raise RuntimeError("Not enough positive samples to train C-VAE (need >= 3).")

dataset = TensorDataset(pos)

# 80/20 split
n_train = max(2, int(0.8 * N))
n_val = max(1, N - n_train)
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=min(batch_size, n_train), shuffle=True)
val_loader = DataLoader(val_ds, batch_size=min(batch_size, n_val), shuffle=False)

# --------------------------------------------------------------
# Build model
# --------------------------------------------------------------
model = GenerativeCVAE(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = float("inf")
best_state = None

history = {
    "train_loss": [], "val_loss": [],
    "train_recon": [], "train_kl": [],
    "val_recon": [], "val_kl": [],
}

# --------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------
for epoch in range(1, epochs + 1):

    # TRAIN
    model.train()
    t_loss = t_recon = t_kl = 0.0

    for (batch,) in train_loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss, recon_loss, kl_loss = vae_loss_function(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()

        t_loss += loss.item() * batch.size(0)
        t_recon += recon_loss * batch.size(0)
        t_kl += kl_loss * batch.size(0)

    train_loss = t_loss / len(train_loader.dataset)
    train_recon = t_recon / len(train_loader.dataset)
    train_kl = t_kl / len(train_loader.dataset)

    history["train_loss"].append(train_loss)
    history["train_recon"].append(train_recon)
    history["train_kl"].append(train_kl)

    # VALIDATION
    model.eval()
    v_loss = v_recon = v_kl = 0.0
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(DEVICE)
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss_function(recon, batch, mu, logvar)

            v_loss += loss.item() * batch.size(0)
            v_recon += recon_loss * batch.size(0)
            v_kl += kl_loss * batch.size(0)

    val_loss = v_loss / len(val_loader.dataset)
    val_recon = v_recon / len(val_loader.dataset)
    val_kl = v_kl / len(val_loader.dataset)

    history["val_loss"].append(val_loss)
    history["val_recon"].append(val_recon)
    history["val_kl"].append(val_kl)

    print(f"Epoch {epoch}/{epochs} | Train={train_loss:.6f} | Val={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = deepcopy(model.state_dict())
        torch.save({"state_dict": best_state, "val_loss": best_val_loss}, MODEL_OUT)
        print(f"  -> New best model saved (val_loss={best_val_loss:.6f})")

# Ensure model saved
if best_state is None:
    best_state = model.state_dict()
    torch.save({"state_dict": best_state, "val_loss": best_val_loss}, MODEL_OUT)

# --------------------------------------------------------------
# Save metrics
# --------------------------------------------------------------
metrics = {
    "best_val_loss": best_val_loss,
    "history": history,
    "config": {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs
    }
}

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

print("C-VAE training complete.")
print(f"Saved best model: {MODEL_OUT}")
print(f"Saved metrics: {METRICS_OUT}")
