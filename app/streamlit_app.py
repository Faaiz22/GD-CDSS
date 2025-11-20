"""
Gene-Drug CDSS v2 - Streamlit Entry Point
FIXED: Better artifact handling with clear setup instructions
"""

import streamlit as st
import sys
import os

# ============================================================================
# PATH RESOLUTION
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
    """Verify required artifacts exist"""
    required = {
        "Core Data (1)": [
            "artifacts/id_maps.json",
        ],
        "Models (2)": [
            "artifacts/model.pt",
            "artifacts/cvae_model.pt",
        ],
        "Feature Libraries (2)": [
            "artifacts/drug_library.npy",
            "artifacts/protein_library.npy",
        ],
        "Featurizers (6)": [
            "artifacts/drug_featurizer_desc_scaler.pkl",
            "artifacts/drug_featurizer_fp_scaler.pkl",
            "artifacts/drug_featurizer_fp_pca.pkl",
            "artifacts/protein_featurizer_physchem_scaler.pkl",
            "artifacts/protein_featurizer_dpc_scaler.pkl",
            "artifacts/protein_featurizer_dpc_pca.pkl",
        ],
        "Optional - Phytochemicals (2)": [
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
# MAIN PAGE
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

# Check artifacts
with st.spinner("Validating system artifacts..."):
    missing = check_artifacts()

if missing:
    st.error("⚠️ **Critical: Missing Artifacts**")
    
    # Show what's missing
    for category, files in missing.items():
        if files:
            with st.expander(f"❌ {category} ({len(files)} missing)", expanded=True):
                for f in files:
                    st.code(f, language="")
    
    st.divider()
    
    # Smart setup instructions based on what exists
    has_colab_csv = any([
        os.path.exists(os.path.join(PROJECT_ROOT, "data", "raw", fname))
        for fname in [
            "Unified_Gene-Drug_Association_Protein_Features.csv",
            "Unified_Gene-Drug_Association_3D_Features.csv",
            "Unified_Gene-Drug_Association_with_Sequences.csv",
        ]
    ])
    
    has_models = not missing.get("Models (2)")
    has_featurizers = not missing.get("Featurizers (6)")
    has_libraries = not missing.get("Feature Libraries (2)")
    
    st.markdown("### 🛠️ How to Fix This")
    
    # ========== SCENARIO 1: Has Colab CSV ==========
    if has_colab_csv:
        st.success("✅ **Detected Colab-generated CSV file!**")
        st.markdown("""
        ### ✨ Quick Setup (Recommended)
        
        You already have the unified CSV from your Colab notebook.
        Just run the artifact generator:
        
        ```bash
        # Step 1: Generate all artifacts from your Colab data
        python scripts/00_generate_from_colab.py
        
        # Step 2: Train models (takes 5-15 minutes)
        python scripts/02_train_association_model.py
        python scripts/03_train_cvae.py
        
        # Step 3: Restart this Streamlit app
        streamlit run streamlit_app.py
        ```
        
        This will create all 13 required artifacts automatically.
        """)
    
    # ========== SCENARIO 2: Has TSV files ==========
    elif os.path.exists(os.path.join(PROJECT_ROOT, "data", "raw", "genes.tsv")):
        st.warning("📋 **Detected raw TSV files (PharmGKB data)**")
        st.markdown("""
        ### Standard Setup
        
        ```bash
        # Step 1: Build artifacts from TSV files (10-20 minutes)
        python scripts/01_build_artifacts.py
        
        # Step 2: Train models
        python scripts/02_train_association_model.py
        python scripts/03_train_cvae.py
        
        # Step 3: Restart app
        streamlit run streamlit_app.py
        ```
        """)
    
    # ========== SCENARIO 3: No data at all ==========
    else:
        st.error("❌ **No source data found**")
        st.markdown("""
        ### Option A: Use Colab Notebook Data (Easiest)
        
        1. **Upload your Colab-generated CSV** to `data/raw/`:
           - `Unified_Gene-Drug_Association_Protein_Features.csv` (best)
           - OR `Unified_Gene-Drug_Association_3D_Features.csv`
           - OR `Unified_Gene-Drug_Association_with_Sequences.csv`
        
        2. **Optional**: Also upload `phytochemicals_new1.csv` for phyto module
        
        3. **Run generator**:
           ```bash
           python scripts/00_generate_from_colab.py
           python scripts/02_train_association_model.py
           python scripts/03_train_cvae.py
           ```
        
        ---
        
        ### Option B: Use PharmGKB Raw Data
        
        1. **Download PharmGKB data** from https://www.pharmgkb.org/downloads
        2. **Place in `data/raw/`**:
           - `genes.tsv`
           - `drugs.tsv`
           - `relationships.tsv`
        
        3. **Build**:
           ```bash
           python scripts/01_build_artifacts.py
           python scripts/02_train_association_model.py
           python scripts/03_train_cvae.py
           ```
        """)
    
    st.divider()
    
    # File checklist
    st.markdown("### 📋 Current File Status")
    
    data_files = [
        "data/raw/Unified_Gene-Drug_Association_Protein_Features.csv",
        "data/raw/genes.tsv",
        "data/raw/drugs.tsv",
        "data/raw/relationships.tsv",
        "data/raw/phytochemicals_new1.csv",
    ]
    
    for fpath in data_files:
        full_path = os.path.join(PROJECT_ROOT, fpath)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            st.success(f"✅ {fpath} ({size_mb:.2f} MB)")
        else:
            st.error(f"❌ {fpath}")
    
    st.stop()

# System ready!
st.success("✅ All artifacts validated. System ready.")

# ============================================================================
# NAVIGATION GUIDE
# ============================================================================

st.markdown("""
### 🚀 Get Started

**Select a module from the sidebar:**
- 🔬 **Prediction & XAI**: Single drug-gene predictions with explainability
- 🔄 **Drug Repurposing**: Screen entire libraries against target genes
- 🧬 **Generative Discovery**: Generate novel drug candidates

---

### 📊 System Capabilities

**1. Prediction & Explainability**
- Live NCBI protein sequence fetching
- 135-dim drug + 38-dim protein feature engineering
- Neural network association prediction
- SHAP waterfall explanations

**2. Drug Repurposing**
- **Phytochemical Library**: 1000+ natural compounds
- **PharmGKB Library**: FDA-approved drugs
- Advanced filtering (GI absorption, BBB penetration)

**3. Generative Discovery**
- Conditional VAE trained on positive associations
- Latent space sampling for novel candidates
- Nearest-neighbor analog identification

---

### 🎯 Quick Test

1. Go to **"Prediction & XAI"** (sidebar)
2. **SMILES**: `CCO` (ethanol)
3. **Gene**: `TP53`
4. **Click "Run Prediction"**

Expected output:
- ✅ Protein sequence fetched (393 amino acids)
- ✅ Association score (0.0-1.0)
- ✅ SHAP waterfall plot

---

### 📝 Citation

```
Gene-Drug CDSS v2: An AI-Powered Clinical Decision Support System
for Personalized Pharmacogenomics (2024)
```

**⚠️ Disclaimer**: Research prototype. Not validated for clinical use.
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
    
    # Artifact status
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
    
    # Quick tips
    st.info("""
    **👈 Pages in Sidebar:**
    - 🔬 Prediction & XAI
    - 🔄 Drug Repurposing  
    - 🧬 Generative Discovery
    
    If pages don't appear, check that `pages/` folder exists with numbered files.
    """)
    
    st.divider()
    
    st.markdown("""
    ### 📚 Resources
    - [GitHub Repo](https://github.com/faaiz22/gd-cdss)
    - [Report Issues](https://github.com/faaiz22/gd-cdss/issues)
    """)
    
    st.caption("Gene-Drug CDSS v2 | Built with Streamlit")
