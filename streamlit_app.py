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
    required = {
        "Core Data": [
            "artifacts/id_maps.json",
        ],
        "Models": [
            "artifacts/model.pt",
            "artifacts/cvae_model.pt",
        ],
        "Feature Libraries": [
            "artifacts/drug_library.npy",
            "artifacts/protein_library.npy",
        ],
        "Featurizers": [
            "artifacts/drug_featurizer_desc_scaler.pkl",
            "artifacts/drug_featurizer_fp_scaler.pkl",
            "artifacts/drug_featurizer_fp_pca.pkl",
            "artifacts/protein_featurizer_physchem_scaler.pkl",
            "artifacts/protein_featurizer_dpc_scaler.pkl",
            "artifacts/protein_featurizer_dpc_pca.pkl",
        ],
        "Optional (Phytochemicals)": [
            "artifacts/phyto_library.npy",
            "artifacts/phyto_metadata.parquet",
        ]
    }
    
    missing = {}
    for category, files in required.items():
        category_missing = []
        for path in files:
            full_path = os.path.join(PROJECT_ROOT, path)
            if not os.path.exists(full_path):
                category_missing.append(path)
        
        if category_missing:
            missing[category] = category_missing
    
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
    
    # Show what's missing by category
    for category, files in missing.items():
        if files:
            with st.expander(f"❌ {category} ({len(files)} missing)"):
                for f in files:
                    st.code(f)
    
    st.divider()
    
    st.markdown("""
    ### 🛠️ How to Fix This
    
    Your artifacts are missing. Choose **ONE** solution:
    
    #### **Option 1: Git LFS** (Recommended if files < 2GB total)
    
    ```bash
    # Install Git LFS
    brew install git-lfs  # macOS
    # OR
    sudo apt-get install git-lfs  # Linux
    
    # Initialize
    git lfs install
    
    # Track large files
    git lfs track "artifacts/*.npy"
    git lfs track "artifacts/*.pt"
    git lfs track "artifacts/*.parquet"
    git lfs track "artifacts/*.pkl"
    
    # Add and push
    git add .gitattributes
    git add artifacts/
    git commit -m "Add artifacts via Git LFS"
    git push
    ```
    
    #### **Option 2: Build Artifacts Now** (Slowest but works)
    
    If you have raw data files in `data/raw/`, you can build artifacts:
    
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt
    
    # 2. Build artifacts (takes 5-15 minutes)
    python scripts/01_build_artifacts.py
    python scripts/02_train_association_model.py
    python scripts/03_train_cvae.py
    
    # 3. Commit and push
    git add artifacts/
    git commit -m "Add generated artifacts"
    git push
    ```
    
    #### **Option 3: External Storage** (Best for > 2GB)
    
    Upload artifacts to Google Drive and modify `src/utils/streamlit_helpers.py`:
    
    ```python
    # Add to requirements.txt:
    gdown==4.7.1
    
    # In streamlit_helpers.py:
    import gdown
    gdown.download_folder(
        id="YOUR_GOOGLE_DRIVE_FOLDER_ID",
        output="artifacts/",
        quiet=False
    )
    ```
    
    ---
    
    ### 📊 Check Your Artifact Sizes
    
    ```bash
    # See which files are too large for GitHub
    find artifacts -type f -size +100M
    
    # Total size
    du -sh artifacts/
    ```
    
    **GitHub limit**: 100MB per file (without Git LFS)
    
    ---
    
    ### ℹ️ What Are These Files?
    
    - **`*.pt`**: Trained PyTorch models (association + C-VAE)
    - **`*.npy`**: Pre-computed feature vectors for drugs/proteins
    - **`*.pkl`**: Trained scalers and PCA models for featurization
    - **`*.parquet`**: Phytochemical metadata
    - **`id_maps.json`**: Gene/drug ID mappings
    
    These files are **generated** by the scripts in `scripts/`, but are too large to include in the repository by default.
    """)
    
    # Show file tree for reference
    with st.expander("📁 Expected File Structure"):
        st.code("""
artifacts/
├── id_maps.json                           (~50 KB)
├── model.pt                                (~2-5 MB)
├── cvae_model.pt                           (~1-3 MB)
├── drug_library.npy                        (~10-50 MB)
├── protein_library.npy                     (~5-20 MB)
├── phyto_library.npy                       (~5-20 MB, optional)
├── phyto_metadata.parquet                  (~1-5 MB, optional)
├── drug_featurizer_desc_scaler.pkl         (~10 KB)
├── drug_featurizer_fp_scaler.pkl           (~100 KB)
├── drug_featurizer_fp_pca.pkl              (~500 KB)
├── protein_featurizer_physchem_scaler.pkl  (~5 KB)
├── protein_featurizer_dpc_scaler.pkl       (~50 KB)
└── protein_featurizer_dpc_pca.pkl          (~200 KB)
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
    
    # Show artifact status
    st.subheader("Artifact Status")
    missing_artifacts = check_artifacts()
    
    if not missing_artifacts:
        st.success("All artifacts loaded ✓")
    else:
        total_missing = sum(len(files) for files in missing_artifacts.values())
        st.error(f"{total_missing} artifacts missing")
        
        for category, files in missing_artifacts.items():
            if files:
                st.caption(f"❌ {category}: {len(files)} missing")
    
    st.divider()
    
    st.markdown("""
    ### 📚 Resources
    - [GitHub Repo](https://github.com/faaiz22/gd-cdss)
    - [Report Issues](https://github.com/faaiz22/gd-cdss/issues)
    """)
    
    st.divider()
    
    st.caption("Gene-Drug CDSS v2 | Built with Streamlit")
