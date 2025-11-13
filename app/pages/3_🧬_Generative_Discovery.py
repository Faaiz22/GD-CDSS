
import sys, os

# NOTE: sys.path is handled in Cell 1 to ensure project root is discoverable.
# The following path resolution logic is for standalone Streamlit deployment only
# and is commented out for Colab to avoid redundant operations and potential conflicts.
# current_script_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# ============================================================================
# IMPORTS (AFTER GLOBAL PATH FIX IN CELL 1)
# ============================================================================
import streamlit as st
import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
# Note: scipy.stats.norm is not used in the current version of the code,
# but it was in a previous version for distribution plotting. Keeping the import
# for now, but it could be removed if not needed.
# from scipy.stats import norm 

from src.utils.streamlit_helpers import (
    load_drug_featurizer,
    load_protein_featurizer,
    load_association_model,
    load_cvae_model,
    load_id_maps,
    load_pharmgkb_drug_library,
    nearest_neighbors,
)

from src.features.sequence_fetcher import (
    get_fasta_from_gene_symbol,
    parse_fasta_to_sequence,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Generative Discovery",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Generative Discovery — C-VAE Molecular Feature Synthesis")

st.markdown("""
This module performs **in-silico generative drug design** using the trained **C-VAE** model.

The system:
1. Samples random latent vectors
2. Decodes them into **135-dim drug feature vectors**
3. Validates generated vectors against the selected gene
4. Finds nearest known PharmGKB drug analogs
5. Displays results in a clear, actionable format

⚠️ **Important**:
This generates *feature vectors*, not SMILES.
Inverse-mapping (vector → SMILES) remains an open research problem.
""")

# ============================================================================
# LOAD MODELS + DATA
# ============================================================================
@st.cache_resource
def load_all_models_and_data():
    drug_feat = load_drug_featurizer()
    prot_feat = load_protein_featurizer()
    assoc_model = load_association_model()
    cvae_model = load_cvae_model()
    id_maps = load_id_maps()
    pharmgkb_vecs = load_pharmgkb_drug_library()
    return drug_feat, prot_feat, assoc_model, cvae_model, id_maps, pharmgkb_vecs

with st.spinner("Loading all models and data..."):
    drug_feat, prot_feat, assoc_model, cvae_model, id_maps, pharmgkb_vecs = load_all_models_and_data()
    device = next(assoc_model.parameters()).device

latent_dim = cvae_model.latent_dim
# drug_dim = assoc_model.drug_dim # This attribute is not directly available on assoc_model
# Get drug_dim from the loaded drug featurizer, or config if needed
dummy_smiles_for_dim = "C"
drug_dim = drug_feat.build_feature_vector(dummy_smiles_for_dim).shape[0]

# ============================================================================
# SELECT TARGET GENE
# ============================================================================
st.subheader("1. Select Target Gene")
gene_symbols = id_maps.get("ordered_symbols_list", [])
target_gene = st.selectbox("Target Gene for Generative Validation:", gene_symbols)

# Store protein vector in session state to persist across reruns
if "target_prot_vec_gen" not in st.session_state:
    st.session_state["target_prot_vec_gen"] = None

fetch_btn = st.button("Fetch Protein for Generation/Validation")

if fetch_btn:
    try:
        with st.spinner(f"Fetching FASTA for {target_gene}..."):
            fasta = get_fasta_from_gene_symbol(target_gene, email="example@example.com")
            seq = parse_fasta_to_sequence(fasta)
            st.session_state["target_prot_vec_gen"] = prot_feat.build_feature_vector(seq)
            st.success(f"Protein sequence fetched for {target_gene} ({len(seq)} aa).")
    except Exception as e:
        st.error(f"Failed to fetch protein sequence for {target_gene}: {e}")
        st.session_state["target_prot_vec_gen"] = None

prot_vec = st.session_state["target_prot_vec_gen"]

if prot_vec is None:
    st.info("Please fetch a target protein to proceed with generation and validation.")
    st.stop()
else:
    prot_vec_t = torch.tensor(prot_vec, dtype=torch.float32).unsqueeze(0).to(device)
    st.markdown(f"**Target protein '{target_gene}' loaded.** Ready for generation.")

# ============================================================================
# GENERATION CONTROLS
# ============================================================================
st.subheader("2. Generate Drug Candidates")

n_generate = st.slider("Number of new drug candidates to generate:", 1, 50, 5)
temperature = st.slider("Sampling Temperature (latent noise):", 0.1, 2.0, 1.0, 0.1)

gen_btn = st.button("Generate New Drugs")

# ============================================================================
# GENERATION PIPELINE
# ============================================================================
if gen_btn:
    # ----------------------------------------------------------------------
    # Step 1 — Sample latent vectors
    # ----------------------------------------------------------------------
    st.markdown("**Step 1: Sampling Latent Vectors**")

    with st.spinner("Sampling latent space..."):
        z = torch.randn((n_generate, latent_dim), device=device) * float(temperature)

    st.markdown(f"_Sampled **{n_generate} latent vectors**, each of dim **{latent_dim}**._")

    # ----------------------------------------------------------------------
    # Step 2 — Decode through C-VAE
    # ----------------------------------------------------------------------
    st.markdown("**Step 2: Decode Latent → Drug Feature Vectors**")

    try:
        cvae_model.eval()
        with torch.no_grad():
            decoded = cvae_model.decode(z)  # shape [n_generate, drug_dim]
            gen_vectors = decoded.cpu().numpy()
            st.session_state["generated_drug_vecs"] = gen_vectors
    except Exception as e:
        st.error(f"Decoding error: {e}")
        st.stop()

    st.markdown(f"_Decoded into feature vectors of dimension **{drug_dim}**._")

    # ----------------------------------------------------------------------
    # Step 3 — Validate using association model
    # ----------------------------------------------------------------------
    st.markdown("**Step 3: Validate Generated Molecules**")

    assoc_model.eval()
    scores = []

    with torch.no_grad():
        prot_batch = prot_vec_t.repeat(n_generate, 1)
        drug_batch = torch.tensor(gen_vectors, dtype=torch.float32).to(device)
        preds = assoc_model(drug_batch, prot_batch).cpu().numpy().ravel()
        scores = preds.tolist()

    results_df = pd.DataFrame({
        "Generated_ID": [f"Gen_{i+1}" for i in range(n_generate)],
        "Predicted_Association_Score": scores
    }).sort_values("Predicted_Association_Score", ascending=False)

    st.dataframe(results_df, use_container_width=True)
    st.session_state["generated_scores_df"] = results_df

    # ----------------------------------------------------------------------
    # Step 4 — Nearest Neighbor Analysis
    # ----------------------------------------------------------------------
    st.markdown("**Step 4: Nearest Known Drug Analogs**")

    pharm_matrix = np.array(pharmgkb_vecs)
    # Ensure pharm_matrix has enough samples for nearest_neighbors
    if pharm_matrix.shape[0] == 0:
        st.warning("PharmGKB drug library is empty. Cannot find nearest neighbors.")
    else:
        # Fix: nearest_neighbors expects a 2D array for query_vecs, even for a single vector
        nn_indices = nearest_neighbors(pharm_matrix, gen_vectors, topk=min(3, pharm_matrix.shape[0]))

        drug_ids = id_maps.get("ordered_drug_ids", [f"Drug_{i}" for i in range(pharm_matrix.shape[0])])

        nn_rows = []
        for i, idx_list in enumerate(nn_indices):
            neighbors = [drug_ids[j] for j in idx_list]
            dists = [np.linalg.norm(gen_vectors[i] - pharm_matrix[j]) for j in idx_list]

            nn_rows.append({
                "Generated_ID": f"Gen_{i+1}",
                "Nearest_Drugs": ", ".join(neighbors),
                "Distances": ", ".join([f"{d:.4f}" for d in dists])
            })

        nn_df = pd.DataFrame(nn_rows)
        st.dataframe(nn_df, use_container_width=True)

    st.success("Generation + validation completed successfully!")


# Display results if available from a previous run (after refresh)
if "generated_scores_df" in st.session_state and st.session_state["generated_scores_df"] is not None:
    st.subheader("Last Generation Results")
    st.dataframe(st.session_state["generated_scores_df"], use_container_width=True)

    # Distribution plot for last generation
    st.subheader("Score Distribution of Generated Candidates (Last Run)")
    drug_library_scores = []
    with torch.no_grad():
        for vec in pharmgkb_vecs:
            dv_t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
            s = float(assoc_model(dv_t, prot_vec_t).cpu().numpy().ravel()[0])
            drug_library_scores.append(s)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=drug_library_scores, name='PharmGKB Drug Library', opacity=0.7, xbins=dict(size=0.05)))
    fig.add_trace(go.Histogram(x=st.session_state["generated_scores_df"]["Predicted_Association_Score"], name='Generated Candidates', opacity=0.7, xbins=dict(size=0.05)))
    fig.update_layout(barmode='overlay', title_text='Distribution of Predicted Association Scores')
    st.plotly_chart(fig, use_container_width=True)
