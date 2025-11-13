
import sys, os

# NOTE: sys.path is handled in Cell 1 to ensure project root is discoverable.
# The following logic is for standalone Streamlit deployment and is commented out for Colab.
# current_script_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


import streamlit as st
import pandas as pd
import numpy as np
import torch

from src.utils.streamlit_helpers import (
    load_drug_featurizer,
    load_protein_featurizer,
    load_association_model,
    load_phyto_metadata,
    load_phyto_library,
    load_pharmgkb_drug_library,
    load_id_maps,
    nearest_neighbors,
)

from src.features.sequence_fetcher import (
    get_fasta_from_gene_symbol,
    parse_fasta_to_sequence,
)

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Drug Repurposing", page_icon="🔄", layout="wide")
st.title("🔄 Drug Repurposing Dashboard")

st.markdown("""
Two repurposing engines:

### 🌿 Phytochemical Library (with filters)
- Family, Class
- Minimum predicted score
- GI absorption (White == 1)
- BBB penetration (Yellow == 1)

### 💊 PharmGKB Drug Library
- Top-ranked drugs by predicted association score
""")

# ============================================================================
# Load Models + Data
# ============================================================================
with st.spinner("Loading models & featurizers..."):
    drug_feat = load_drug_featurizer()
    prot_feat = load_protein_featurizer()
    assoc_model = load_association_model()
    id_maps = load_id_maps()

with st.spinner("Loading libraries..."):
    phyto_meta = load_phyto_metadata()
    phyto_vecs = load_phyto_library()
    pharmgkb_vecs = load_pharmgkb_drug_library()

# ============================================================================
# Select Gene Target (required for scoring)
# ============================================================================
gene_symbols = id_maps["ordered_symbols_list"]
gene_symbol = st.selectbox("Target Gene for Repurposing:", gene_symbols)

# ============================================================================
# Fetch protein feature vector for target gene
# ============================================================================
fetch_btn = st.button("Fetch Target Protein")
prot_vec = None

if fetch_btn:
    try:
        fasta = get_fasta_from_gene_symbol(gene_symbol, email="example@example.com")
        seq = parse_fasta_to_sequence(fasta)
        prot_vec = prot_feat.build_feature_vector(seq)
        st.success(f"Protein sequence fetched ({len(seq)} aa).")
    except Exception as e:
        st.error(f"Failed to fetch protein sequence: {e}")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2 = st.tabs(["🌿 Phytochemicals", "💊 PharmGKB Drugs"])

# ============================================================================
# TAB 1: PHYTOS
# ============================================================================
with tab1:

    st.subheader("🌿 Phytochemical Library")

    if prot_vec is None:
        st.info("Fetch target protein first.")
        st.stop()

    # filters
    st.sidebar.header("Phytochemical Filters")

    families = sorted(phyto_meta["Family"].dropna().unique())
    selected_family = st.sidebar.selectbox("Family:", ["All"] + families)

    class_col = "Class "
    classes = sorted(phyto_meta[class_col].dropna().unique())
    selected_class = st.sidebar.selectbox("Class:", ["All"] + classes)

    min_score = st.sidebar.slider("Minimum Score:", 0.0, 1.0, 0.0, 0.01)
    filter_hia = st.sidebar.toggle("High GI Absorption (White == 1)")
    filter_bbb = st.sidebar.toggle("BBB Penetrant (Yellow == 1)")

    # score all phytos
    device = next(assoc_model.parameters()).device
    prot_vec_t = torch.tensor(prot_vec, dtype=torch.float32).unsqueeze(0).to(device)

    st.write("Scoring phytochemicals...")

    scores = []
    with torch.no_grad():
        for vec in phyto_vecs:
            dv = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
            s = float(assoc_model(dv, prot_vec_t).cpu().numpy().ravel()[0])
            scores.append(s)

    df = phyto_meta.copy()
    df["Predicted_Association_Score"] = scores

    # apply filters
    if selected_family != "All":
        df = df[df["Family"] == selected_family]

    if selected_class != "All":
        df = df[df[class_col] == selected_class]

    df = df[df["Predicted_Association_Score"] >= min_score]

    if filter_hia:
        df = df[df["White"] == 1]

    if filter_bbb:
        df = df[df["Yellow"] == 1]

    df = df.sort_values("Predicted_Association_Score", ascending=False)
    st.dataframe(df, use_container_width=True)

# ============================================================================
# TAB 2: PHARMGKB DRUGS
# ============================================================================
with tab2:

    st.subheader("💊 PharmGKB Drug Library")

    if prot_vec is None:
        st.info("Fetch target protein first.")
        st.stop()

    scores = []
    with torch.no_grad():
        prot_vec_t = torch.tensor(prot_vec, dtype=torch.float32).unsqueeze(0).to(device)
        for vec in pharmgkb_vecs:
            dv = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
            s = float(assoc_model(dv, prot_vec_t).cpu().numpy().ravel()[0])
            scores.append(s)

    drug_ids = id_maps["ordered_drug_ids"]

    df = pd.DataFrame({
        "Drug_ID": drug_ids,
        "Predicted_Association_Score": scores
    })

    df = df.sort_values("Predicted_Association_Score", ascending=False).head(50)

    st.dataframe(df, use_container_width=True)
