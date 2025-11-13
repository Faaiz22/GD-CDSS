import sys, os

# Ensure repo root + src/ are on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import yaml

from src.features.drug_featurizer import DrugFeaturizer
from src.features.protein_featurizer import ProteinFeaturizer
from src.features.sequence_fetcher import (
    get_fasta_from_gene_symbol,
    parse_fasta_to_sequence,
)

# Load config
with open(os.path.join(ROOT, "config", "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

C = CFG["column_names"]
P = CFG["pipeline"]
A = CFG["artifacts"]

RAW_GENES = CFG["raw_data"]["genes_tsv"]
RAW_DRUGS = CFG["raw_data"]["drugs_tsv"]
RAW_RELS  = CFG["raw_data"]["relationships_tsv"]
RAW_PHYTO = CFG["raw_data"]["phytochemicals_csv"]

os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
ART = os.path.join(ROOT, "artifacts")

print("========== BUILDING ARTIFACTS ==========")

# 1. Load genes
genes = pd.read_csv(RAW_GENES, sep="\t", dtype=str)
genes = genes.dropna(subset=[C["gene_id"], C["gene_symbol"]])
genes = genes.sort_values(C["gene_id"])

ordered_gene_ids = genes[C["gene_id"]].tolist()
ordered_gene_symbols = genes[C["gene_symbol"]].tolist()

gene_id_to_index = {gid: i for i, gid in enumerate(ordered_gene_ids)}
gene_id_to_symbol = dict(zip(ordered_gene_ids, ordered_gene_symbols))

print(f"Loaded {len(genes)} genes.")

# 2. Load drugs
drugs = pd.read_csv(RAW_DRUGS, sep="\t", dtype=str)
drugs = drugs.dropna(subset=[C["drug_id"], C["drug_smiles"]])
drugs = drugs.sort_values(C["drug_id"])

ordered_drug_ids = drugs[C["drug_id"]].tolist()
ordered_drug_smiles = drugs[C["drug_smiles"]].tolist()

drug_id_to_index = {did: i for i, did in enumerate(ordered_drug_ids)}

print(f"Loaded {len(drugs)} drugs.")

# 3. Drug Featurizer
drug_feat = DrugFeaturizer()
drug_feat.fit_descriptor_scaler(ordered_drug_smiles)
drug_feat.fit_fp_pca(ordered_drug_smiles, n_components=128)

drug_vectors = np.vstack([
    drug_feat.build_feature_vector(smi)
    for smi in tqdm(ordered_drug_smiles, desc="Drug Vectors")
])
np.save(os.path.join(ART, "drug_library.npy"), drug_vectors)

drug_feat.save(os.path.join(ART, "drug_featurizer"))
print("Drug vectors saved.")

# 4. Protein Featurizer
protein_sequences = []
for sym in tqdm(ordered_gene_symbols, desc="Fetching FASTA"):
    try:
        fasta = get_fasta_from_gene_symbol(sym)
        seq = parse_fasta_to_sequence(fasta)
    except:
        seq = ""
    protein_sequences.append(seq)

prot_feat = ProteinFeaturizer()
prot_feat.fit_dpc_pca(protein_sequences)
prot_feat.fit_physchem_scaler(protein_sequences)

protein_vectors = np.vstack([
    prot_feat.build_feature_vector(seq)
    for seq in tqdm(protein_sequences, desc="Protein vectors")
])
np.save(os.path.join(ART, "protein_library.npy"), protein_vectors)

prot_feat.save(os.path.join(ART, "protein_featurizer"))
print("Protein vectors saved.")

# 5. Relationships → association_dataset + positive pairs
rels = pd.read_csv(RAW_RELS, sep="\t", dtype=str)

X_drug, X_prot, Y = [], [], []
positive_pairs = []

for _, row in rels.iterrows():
    if row[C["entity1_type"]] != "Gene":
        continue
    if row[C["entity2_type"]] != "Chemical":
        continue

    gid = row[C["entity1_id"]]
    did = row[C["entity2_id"]]

    if gid not in gene_id_to_index or did not in drug_id_to_index:
        continue

    gidx = gene_id_to_index[gid]
    didx = drug_id_to_index[did]

    dv = drug_vectors[didx]
    pv = protein_vectors[gidx]

    label = 1.0 if str(row[C["association_label"]]).strip().lower() == "associated" else 0.0

    X_drug.append(dv)
    X_prot.append(pv)
    Y.append(label)

    if label == 1.0:
        positive_pairs.append(dv)

X_drug = torch.tensor(np.vstack(X_drug), dtype=torch.float32)
X_prot = torch.tensor(np.vstack(X_prot), dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

torch.save({"drug": X_drug, "prot": X_prot, "label": Y}, os.path.join(ART, "association_dataset.pt"))
torch.save(torch.tensor(np.vstack(positive_pairs), dtype=torch.float32), os.path.join(ART, "positive_feature_pairs.pt"))

print(f"Saved {len(Y)} association rows.")
print(f"Saved {len(positive_pairs)} positive pairs.")

# 6. Phytochemicals - auto-detect SMILES
phy = pd.read_csv(RAW_PHYTO, dtype=str)

import re
re_smi = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)\\\/%=#$]+$")

def is_smiles(s):
    if not isinstance(s, str): return False
    return bool(re_smi.match(s.strip()))

candidates = []
for col in phy.columns:
    vals = phy[col].dropna().astype(str)
    valid = vals.apply(is_smiles).sum()
    if valid >= 5 and valid / max(1, len(vals)) > 0.3:
        candidates.append((col, valid))

if not candidates:
    raise KeyError(f"No SMILES column detected. Columns: {list(phy.columns)}")

smiles_col = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
print("Detected SMILES column:", smiles_col)

phy = phy.dropna(subset=[smiles_col])
phyto_smiles = phy[smiles_col].astype(str).tolist()

phyto_vectors = []
for smi in tqdm(phyto_smiles, desc="Phytochemical Featurization"):
    try:
        phyto_vectors.append(drug_feat.build_feature_vector(smi))
    except:
        phyto_vectors.append(np.zeros(P["drug_vector_dim"]))

phyto_vectors = np.vstack(phyto_vectors)
np.save(os.path.join(ART, "phyto_library.npy"), phyto_vectors)
phy.to_parquet(os.path.join(ART, "phyto_metadata.parquet"), index=False)

# 7. ID Maps
id_maps = {
    "ordered_gene_ids": ordered_gene_ids,
    "ordered_symbols_list": ordered_gene_symbols,
    "ordered_drug_ids": ordered_drug_ids,
    "gene_id_to_index": gene_id_to_index,
    "drug_id_to_index": drug_id_to_index,
    "gene_id_to_symbol": gene_id_to_symbol,
}
with open(os.path.join(ART, "id_maps.json"), "w") as f:
    json.dump(id_maps, f, indent=2)

# 8. Feature metadata
meta = {
    "drug_vector_dim": P["drug_vector_dim"],
    "protein_vector_dim": P["protein_vector_dim"],
}
with open(os.path.join(ART, "feature_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("========== ARTIFACT BUILD COMPLETE ==========")
