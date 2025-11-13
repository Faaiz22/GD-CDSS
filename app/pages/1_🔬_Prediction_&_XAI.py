
import sys
import os

# NOTE: sys.path is handled in Cell 1 to ensure project root is discoverable.
# The following logic is for standalone Streamlit deployment and is commented out for Colab.
# current_script_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


import streamlit as st
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

# FIXED IMPORTS — Streamlit pages execute inside gene_drug_cdss_v2/app/
from src.utils.streamlit_helpers import (
    load_drug_featurizer,
    load_protein_featurizer,
    load_association_model,
    get_ordered_gene_symbols
)

from src.features.sequence_fetcher import (
    get_fasta_from_gene_symbol,
    parse_fasta_to_sequence
)

# ----------------------------------------------
# STREAMLIT UI
# ----------------------------------------------
st.set_page_config(page_title="Prediction & XAI", page_icon="🔬", layout="wide")

st.title("🔬 Gene–Drug Association Prediction & XAI")

st.markdown("""
Provide **SMILES** and a **Gene Symbol**.
System performs:
1. Live protein FASTA fetch
2. Drug + protein featurization
3. Prediction
4. SHAP waterfall explanation
""")

# Inputs
smiles = st.text_input("SMILES:", "")
symbols = get_ordered_gene_symbols()
gene_symbol = st.selectbox("Gene Symbol:", symbols)

email = st.text_input("NCBI Email:", "example@example.com")

run_btn = st.button("Run Prediction")

# ----------------------------------------------
# Run
# ----------------------------------------------
if run_btn:

    if not smiles.strip():
        st.error("SMILES cannot be empty.")
        st.stop()

    with st.spinner("Loading models..."):
        drug_feat = load_drug_featurizer()
        prot_feat = load_protein_featurizer()
        model = load_association_model()

    # Fetch FASTA
    try:
        fasta = get_fasta_from_gene_symbol(gene_symbol, email=email)
        seq = parse_fasta_to_sequence(fasta)
    except Exception as e:
        st.error(f"FASTA fetch failed: {e}")
        st.stop()

    # Featurization
    try:
        dv = drug_feat.build_feature_vector(smiles)
        pv = prot_feat.build_feature_vector(seq)
    except Exception as e:
        st.error(f"Featurization failed: {e}")
        st.stop()

    dv_t = torch.tensor(dv, dtype=torch.float32).unsqueeze(0)
    pv_t = torch.tensor(pv, dtype=torch.float32).unsqueeze(0)

    device = next(model.parameters()).device
    dv_t = dv_t.to(device)
    pv_t = pv_t.to(device)

    with torch.no_grad():
        score = float(model(dv_t, pv_t).cpu().numpy().ravel()[0])

    st.metric("Association Score", f"{score:.4f}")

    # ----------------------------------------------
    # SHAP XAI
    # ----------------------------------------------
    st.subheader("SHAP Explanation")

    try:
        combined = np.concatenate([dv, pv]).reshape(1, -1)

        def f(X):
            X = torch.tensor(X, dtype=torch.float32, device=device)
            d = X[:, :135]
            p = X[:, 135:]
            with torch.no_grad():
                o = model(d, p).cpu().numpy()
            return o

        explainer = shap.KernelExplainer(f, combined)
        shap_vals = explainer.shap_values(combined)[0]

        drug_names = [f"DESC_{i}" for i in range(1, 8)] + \
                     [f"FP_PCA_{i}" for i in range(1, 129)]
        prot_names = [f"KMER_PCA_{i}" for i in range(1, 32)] + \
                     [f"PHYSCHEM_{i}" for i in range(1, 8)]
        feature_names = drug_names + prot_names

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals,
                base_values=np.array([0.0]),
                data=combined[0],
                feature_names=feature_names
            ),
            max_display=25,
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP failed: {e}")
