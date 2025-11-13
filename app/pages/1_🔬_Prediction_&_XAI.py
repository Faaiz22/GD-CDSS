"""
Prediction & XAI Module - PRODUCTION READY
No path hacks - relies on streamlit_app.py setting sys.path
"""

import streamlit as st
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import time

from src.utils.streamlit_helpers import (
    load_drug_featurizer,
    load_protein_featurizer,
    load_association_model,
    get_ordered_gene_symbols,
    get_ncbi_email,
)

from src.features.sequence_fetcher import (
    get_fasta_from_gene_symbol,
    parse_fasta_to_sequence
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Prediction & XAI", page_icon="🔬", layout="wide")

st.title("🔬 Gene–Drug Association Prediction & XAI")

st.markdown("""
Predict association strength between any drug (SMILES) and gene target.

**Workflow:**
1. Enter drug SMILES
2. Select target gene
3. System fetches protein sequence from NCBI
4. Featurization (Drug: 135-dim, Protein: 38-dim)
5. Neural network prediction
6. SHAP waterfall explanation

---
""")

# ============================================================================
# RATE LIMITING HELPER
# ============================================================================

def check_rate_limit():
    """Simple rate limiter for NCBI API (3 requests per second limit)"""
    if "last_ncbi_call" not in st.session_state:
        st.session_state.last_ncbi_call = 0
    
    time_since_last = time.time() - st.session_state.last_ncbi_call
    if time_since_last < 0.34:  # 3 requests/sec = 0.33s between requests
        time.sleep(0.34 - time_since_last)
    
    st.session_state.last_ncbi_call = time.time()

# ============================================================================
# INPUTS
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    smiles = st.text_input(
        "Drug SMILES:",
        placeholder="CCO (ethanol example)",
        help="Simplified Molecular Input Line Entry System notation"
    )
    
    st.caption("Examples: CCO (ethanol), CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)")

with col2:
    symbols = get_ordered_gene_symbols()
    
    if not symbols:
        st.error("Gene list not available. Check artifacts.")
        st.stop()
    
    gene_symbol = st.selectbox(
        "Target Gene:",
        symbols,
        help="Select gene target for prediction"
    )

# Email configuration
email = get_ncbi_email()

# ============================================================================
# PREDICTION PIPELINE
# ============================================================================

run_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

if run_btn:
    
    if not smiles.strip():
        st.error("⚠️ SMILES cannot be empty.")
        st.stop()
    
    # ------------------------------------------------------------------------
    # STEP 1: Load Models
    # ------------------------------------------------------------------------
    with st.spinner("Loading models..."):
        try:
            drug_feat = load_drug_featurizer()
            prot_feat = load_protein_featurizer()
            model = load_association_model()
            device = next(model.parameters()).device
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.stop()
    
    st.success("✅ Models loaded")
    
    # ------------------------------------------------------------------------
    # STEP 2: Fetch Protein Sequence
    # ------------------------------------------------------------------------
    with st.spinner(f"Fetching protein sequence for {gene_symbol} from NCBI..."):
        try:
            check_rate_limit()  # Respect NCBI rate limits
            fasta = get_fasta_from_gene_symbol(gene_symbol, email=email)
            seq = parse_fasta_to_sequence(fasta)
            
            st.success(f"✅ Protein fetched: {len(seq)} amino acids")
            
            with st.expander("View Protein Sequence"):
                st.code(seq[:200] + ("..." if len(seq) > 200 else ""))
        
        except Exception as e:
            st.error(f"❌ NCBI fetch failed: {e}")
            st.markdown("""
            **Troubleshooting:**
            - Check gene symbol spelling
            - Verify internet connection
            - NCBI may be rate-limiting (wait 30s and retry)
            - Configure valid email in Streamlit Secrets
            """)
            st.stop()
    
    # ------------------------------------------------------------------------
    # STEP 3: Featurization
    # ------------------------------------------------------------------------
    with st.spinner("Featurizing drug and protein..."):
        try:
            dv = drug_feat.build_feature_vector(smiles)
            pv = prot_feat.build_feature_vector(seq)
            
            st.success(f"✅ Features: Drug [{dv.shape[0]}], Protein [{pv.shape[0]}]")
        
        except Exception as e:
            st.error(f"❌ Featurization failed: {e}")
            st.markdown("**Likely cause:** Invalid SMILES structure")
            st.stop()
    
    # ------------------------------------------------------------------------
    # STEP 4: Prediction
    # ------------------------------------------------------------------------
    with st.spinner("Running neural network prediction..."):
        try:
            dv_t = torch.tensor(dv, dtype=torch.float32).unsqueeze(0).to(device)
            pv_t = torch.tensor(pv, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                score = float(model(dv_t, pv_t).cpu().numpy().ravel()[0])
            
            st.success("✅ Prediction complete")
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()
    
    # ------------------------------------------------------------------------
    # DISPLAY RESULT
    # ------------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Association Score", f"{score:.4f}")
    
    with col2:
        interpretation = (
            "Strong" if score > 0.7 else
            "Moderate" if score > 0.4 else
            "Weak"
        )
        st.metric("Interpretation", interpretation)
    
    with col3:
        confidence = (
            "High" if score > 0.8 or score < 0.2 else
            "Medium"
        )
        st.metric("Confidence", confidence)
    
    # Progress bar visualization
    st.progress(score, text=f"Association Strength: {score:.1%}")
    
    # ------------------------------------------------------------------------
    # STEP 5: SHAP XAI
    # ------------------------------------------------------------------------
    st.divider()
    st.subheader("🔍 Explainability Analysis (SHAP)")
    
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** shows which features contributed most to this prediction.
    - **Red bars**: Features pushing prediction higher
    - **Blue bars**: Features pulling prediction lower
    """)
    
    with st.spinner("Computing SHAP values (this may take 30-60 seconds)..."):
        try:
            # Combine features
            combined = np.concatenate([dv, pv]).reshape(1, -1)
            
            # Wrapper function for SHAP
            def f(X):
                X = torch.tensor(X, dtype=torch.float32, device=device)
                d = X[:, :135]  # Drug features
                p = X[:, 135:]  # Protein features
                with torch.no_grad():
                    o = model(d, p).cpu().numpy()
                return o
            
            # SHAP explainer (use fewer samples for speed)
            explainer = shap.KernelExplainer(f, combined, link="identity")
            shap_vals = explainer.shap_values(combined, nsamples=100)[0]
            
            # Feature names
            drug_names = [f"Drug_DESC_{i}" for i in range(1, 8)] + \
                         [f"Drug_FP_PCA_{i}" for i in range(1, 129)]
            prot_names = [f"Prot_KMER_PCA_{i}" for i in range(1, 32)] + \
                         [f"Prot_PHYSCHEM_{i}" for i in range(1, 8)]
            feature_names = drug_names + prot_names
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals,
                    base_values=explainer.expected_value,
                    data=combined[0],
                    feature_names=feature_names
                ),
                max_display=20,
                show=False
            )
            
            st.pyplot(fig)
            
            st.success("✅ SHAP analysis complete")
            
            # Feature importance table
            with st.expander("📋 Feature Importance Table"):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_vals,
                    'Absolute Importance': np.abs(shap_vals)
                }).sort_values('Absolute Importance', ascending=False).head(15)
                
                st.dataframe(importance_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ SHAP analysis failed: {e}")
            st.markdown("""
            **Note:** SHAP can be computationally expensive. If this fails:
            - Try reducing `nsamples` parameter
            - Use a simpler background dataset
            - Consider alternative explainability methods (LIME, gradient-based)
            """)
    
    # ------------------------------------------------------------------------
    # INTERPRETATION GUIDE
    # ------------------------------------------------------------------------
    st.divider()
    
    with st.expander("📖 How to Interpret Results"):
        st.markdown("""
        ### Association Score
        - **0.0 - 0.3**: Unlikely association (no evidence)
        - **0.3 - 0.5**: Weak association (preliminary signal)
        - **0.5 - 0.7**: Moderate association (worth investigating)
        - **0.7 - 1.0**: Strong association (high confidence)
        
        ### SHAP Values
        - Each bar shows one feature's contribution
        - **Positive (red)**: Increases predicted association
        - **Negative (blue)**: Decreases predicted association
        - **Longer bars**: Stronger influence
        
        ### Drug Features
        - **DESC_1-7**: Molecular descriptors (weight, logP, H-bonds, etc.)
        - **FP_PCA_1-128**: Structural fingerprint components
        
        ### Protein Features
        - **KMER_PCA_1-31**: Dipeptide composition patterns
        - **PHYSCHEM_1-7**: Physical/chemical properties (MW, hydrophobicity, etc.)
        
        ### ⚠️ Limitations
        This is a **research model** trained on PharmGKB data:
        - Not validated for clinical decision-making
        - Predictions require experimental validation
        - Novel drug-gene pairs may be less reliable
        - Does not account for dosage, metabolism, or patient-specific factors
        """)

# ============================================================================
# SIDEBAR INFO
# ============================================================================
with st.sidebar:
    st.header("About This Module")
    
    st.markdown("""
    ### Model Architecture
    - **Type**: Dual-Branch Neural Network
    - **Drug Input**: 135 dimensions
    - **Protein Input**: 38 dimensions
    - **Output**: Association score [0,1]
    
    ### Training Data
    - **Source**: PharmGKB
    - **Samples**: ~10K gene-drug pairs
    - **Labels**: Known associations
    
    ### Performance
    - **AUROC**: ~0.85 (test set)
    - **AUPRC**: ~0.78
    """)
    
    st.divider()
    
    st.caption("💡 Tip: Use the Drug Repurposing module to screen entire libraries at once.")
