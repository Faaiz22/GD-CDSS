"""
Gene-Drug CDSS v2 - Streamlit Cloud Entry Point
CRITICAL: This file MUST be at project root for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# ============================================================================
# PATH RESOLUTION - WORKS ON BOTH LOCAL AND STREAMLIT CLOUD
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Gene-Drug CDSS v2",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ARTIFACT VALIDATION
# ============================================================================
@st.cache_data
def check_artifacts():
    """Verify all required artifacts exist before running"""
    required = [
        "artifacts/id_maps.json",
        "artifacts/drug_library.npy",
        "artifacts/protein_library.npy",
        "artifacts/association_dataset.pt",
        "artifacts/model.pt",
        "artifacts/cvae_model.pt",
        "artifacts/phyto_library.npy",
        "artifacts/phyto_metadata.parquet",
    ]
    
    # Check featurizer models
    required += [
        "artifacts/drug_featurizer_desc_scaler.pkl",
        "artifacts/drug_featurizer_fp_scaler.pkl",
        "artifacts/drug_featurizer_fp_pca.pkl",
        "artifacts/protein_featurizer_physchem_scaler.pkl",
        "artifacts/protein_featurizer_dpc_scaler.pkl",
        "artifacts/protein_featurizer_dpc_pca.pkl",
    ]
    
    missing = []
    for path in required:
        full_path = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(full_path):
            missing.append(path)
    
    return missing

# ============================================================================
# MAIN PAGE CONTENT
# ============================================================================

st.title("🧬 Gene-Drug Clinical Decision Support System v2")

st.markdown("""
## Advanced AI-Powered Drug Discovery Platform

This system integrates:
- **🔬 Prediction & XAI**: Real-time gene-drug association scoring with SHAP explanations
- **🔄 Drug Repurposing**: Phytochemical and PharmGKB drug library screening
- **🧬 Generative Discovery**: C-VAE-based *de novo* molecular feature synthesis

---
""")

# Check artifacts before allowing navigation
with st.spinner("Validating system artifacts..."):
    missing = check_artifacts()

if missing:
    st.error("⚠️ **Critical: Missing Artifacts**")
    st.markdown("""
    The following required files are missing:
    """)
    for m in missing:
        st.code(m)
    
    st.markdown("""
    ### 🛠️ Setup Instructions
    
    **For Local Development:**
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt
    
    # 2. Build artifacts
    python scripts/01_build_artifacts.py
    python scripts/02_train_association_model.py
    python scripts/03_train_cvae.py
    
    # 3. Launch app
    streamlit run streamlit_app.py
    ```
    
    **For Streamlit Cloud Deployment:**
    
    Artifacts are too large for GitHub (>100MB). Use one of these solutions:
    
    1. **Git LFS** (Large File Storage):
       ```bash
       git lfs install
       git lfs track "artifacts/*.npy"
       git lfs track "artifacts/*.pt"
       git lfs track "artifacts/*.parquet"
       ```
    
    2. **External Storage** (Recommended):
       - Upload artifacts to Google Drive/Dropbox/S3
       - Download on first run using `streamlit_helpers.py`
       - Cache with `@st.cache_resource`
    
    3. **On-Demand Generation**:
       - Build lightweight artifacts on Streamlit Cloud startup
       - Trade deployment time for reduced storage
    """)
    
    st.stop()

# System ready
st.success("✅ All artifacts validated. System ready.")

st.markdown("""
### 📊 System Capabilities

**1. Prediction & Explainability**
- Live NCBI protein sequence fetching
- 135-dim drug + 38-dim protein feature engineering
- Neural network association prediction
- SHAP waterfall explanations for model transparency

**2. Drug Repurposing**
- **Phytochemical Library**: 1000+ natural compounds with bioavailability filters
- **PharmGKB Library**: FDA-approved drugs with known gene associations
- Multi-criteria filtering (GI absorption, BBB penetration, molecular families)

**3. Generative Discovery**
- Conditional VAE trained on positive associations
- Latent space sampling for novel drug candidates
- Nearest-neighbor analog identification
- Distribution analysis vs. known drug libraries

---

### 🚀 Navigation

Use the **sidebar** to access different modules.

### ⚙️ Configuration

- **NCBI Email**: Configure in Streamlit Secrets for production
- **Model Hyperparameters**: Edit `config/config.yaml`
- **Data Sources**: Place raw data in `data/raw/`

### 📝 Citation

If you use this system in research, please cite:
```
Gene-Drug CDSS v2: An AI-Powered Clinical Decision Support System
for Personalized Pharmacogenomics (2024)
```

---

**⚠️ Disclaimer**: This is a research prototype. Not validated for clinical use.
""")

# ============================================================================
# SIDEBAR INFO
# ============================================================================
with st.sidebar:
    st.header("System Info")
    
    st.metric("Drug Vector Dim", "135")
    st.metric("Protein Vector Dim", "38")
    st.metric("Association Model", "Dual-Branch NN")
    st.metric("Generative Model", "C-VAE")
    
    st.divider()
    
    st.markdown("""
    ### 📚 Resources
    - [Documentation](#)
    - [GitHub Repo](#)
    - [Report Issues](#)
    """)
    
    st.divider()
    
    st.caption("Gene-Drug CDSS v2 | Built with Streamlit")
