"""
Streamlit Helper Functions - Production Ready
Handles artifact loading with proper error handling and caching
"""

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_distances

# ============================================================================
# PATH MANAGEMENT
# ============================================================================

def get_project_root():
    """Get absolute path to project root"""
    import sys
    # streamlit_app.py is at root
    return os.path.dirname(os.path.abspath(__file__ + "/../../.."))

PROJECT_ROOT = get_project_root()
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# ============================================================================
# ARTIFACT DOWNLOAD (for Streamlit Cloud deployment)
# ============================================================================

@st.cache_resource
def download_artifacts_if_needed():
    """
    Download artifacts from external storage on first run.
    Only needed if using external hosting (Google Drive, S3, etc.)
    """
    # Example: Download from Google Drive
    # This is a template - implement based on your storage solution
    
    required_files = [
        "id_maps.json",
        "drug_library.npy",
        "protein_library.npy",
        "model.pt",
        "cvae_model.pt",
    ]
    
    all_exist = all(
        os.path.exists(os.path.join(ARTIFACTS_DIR, f)) 
        for f in required_files
    )
    
    if not all_exist:
        st.warning("⚠️ Artifacts not found. Attempting download from external storage...")
        
        # IMPLEMENT YOUR DOWNLOAD LOGIC HERE
        # Example for Google Drive:
        # import gdown
        # gdown.download_folder(
        #     id="YOUR_GOOGLE_DRIVE_FOLDER_ID",
        #     output=ARTIFACTS_DIR,
        #     quiet=False
        # )
        
        # For now, show error
        st.error("""
        Artifacts not found locally and auto-download not configured.
        
        Please either:
        1. Build artifacts locally: `python scripts/01_build_artifacts.py`
        2. Configure external storage in `streamlit_helpers.py`
        3. Use Git LFS for large files
        """)
        st.stop()

# ============================================================================
# FEATURIZERS
# ============================================================================

@st.cache_resource
def load_drug_featurizer():
    """Load trained drug featurizer with error handling"""
    try:
        from src.features.drug_featurizer import DrugFeaturizer
        
        feat = DrugFeaturizer()
        path_prefix = os.path.join(ARTIFACTS_DIR, "drug_featurizer")
        
        if not os.path.exists(f"{path_prefix}_desc_scaler.pkl"):
            raise FileNotFoundError(
                f"Drug featurizer models not found at {path_prefix}*"
            )
        
        feat.load(path_prefix)
        return feat
    
    except Exception as e:
        st.error(f"Failed to load drug featurizer: {e}")
        st.stop()


@st.cache_resource
def load_protein_featurizer():
    """Load trained protein featurizer with error handling"""
    try:
        from src.features.protein_featurizer import ProteinFeaturizer
        
        feat = ProteinFeaturizer()
        path_prefix = os.path.join(ARTIFACTS_DIR, "protein_featurizer")
        
        if not os.path.exists(f"{path_prefix}_physchem_scaler.pkl"):
            raise FileNotFoundError(
                f"Protein featurizer models not found at {path_prefix}*"
            )
        
        feat.load(path_prefix)
        return feat
    
    except Exception as e:
        st.error(f"Failed to load protein featurizer: {e}")
        st.stop()

# ============================================================================
# MODELS
# ============================================================================

@st.cache_resource
def load_association_model():
    """Load trained association prediction model"""
    try:
        from src.models.dual_branch_net import DualBranchNet
        
        model_path = os.path.join(ARTIFACTS_DIR, "model.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load config for model dimensions
        import yaml
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        drug_dim = int(cfg["pipeline"]["drug_vector_dim"])
        prot_dim = int(cfg["pipeline"]["protein_vector_dim"])
        hidden = int(cfg["models"]["association_model"]["hidden_dim"])
        dropout = float(cfg["models"]["association_model"]["dropout"])
        
        model = DualBranchNet(
            drug_dim=drug_dim,
            prot_dim=prot_dim,
            hidden_dim=hidden,
            dropout=dropout
        )
        
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return model
    
    except Exception as e:
        st.error(f"Failed to load association model: {e}")
        st.stop()


@st.cache_resource
def load_cvae_model():
    """Load trained C-VAE generative model"""
    try:
        from src.models.generative_cvae import GenerativeCVAE
        
        model_path = os.path.join(ARTIFACTS_DIR, "cvae_model.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"C-VAE model not found at {model_path}")
        
        # Load config
        import yaml
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        input_dim = int(cfg["pipeline"]["drug_vector_dim"])
        latent_dim = int(cfg["models"]["cvae"]["latent_dim"])
        hidden_dim = int(cfg["models"]["cvae"]["hidden_dim"])
        
        model = GenerativeCVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return model
    
    except Exception as e:
        st.error(f"Failed to load C-VAE model: {e}")
        st.stop()

# ============================================================================
# DATA LIBRARIES
# ============================================================================

@st.cache_data
def load_id_maps():
    """Load gene/drug ID mappings"""
    try:
        path = os.path.join(ARTIFACTS_DIR, "id_maps.json")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"ID maps not found at {path}")
        
        with open(path, "r") as f:
            return json.load(f)
    
    except Exception as e:
        st.error(f"Failed to load ID maps: {e}")
        st.stop()


@st.cache_data
def get_ordered_gene_symbols():
    """Quick access to gene symbol list"""
    id_maps = load_id_maps()
    return id_maps.get("ordered_symbols_list", [])


@st.cache_data
def load_pharmgkb_drug_library():
    """Load pre-computed PharmGKB drug feature vectors"""
    try:
        path = os.path.join(ARTIFACTS_DIR, "drug_library.npy")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Drug library not found at {path}")
        
        return np.load(path)
    
    except Exception as e:
        st.error(f"Failed to load PharmGKB drug library: {e}")
        st.stop()


@st.cache_data
def load_phyto_library():
    """Load pre-computed phytochemical feature vectors"""
    try:
        path = os.path.join(ARTIFACTS_DIR, "phyto_library.npy")
        
        if not os.path.exists(path):
            st.warning("Phytochemical library not found. This module will be disabled.")
            return np.array([])  # Return empty array instead of crashing
        
        return np.load(path)
    
    except Exception as e:
        st.warning(f"Failed to load phytochemical library: {e}")
        return np.array([])


@st.cache_data
def load_phyto_metadata():
    """Load phytochemical metadata (names, families, properties)"""
    try:
        path = os.path.join(ARTIFACTS_DIR, "phyto_metadata.parquet")
        
        if not os.path.exists(path):
            st.warning("Phytochemical metadata not found.")
            return pd.DataFrame()  # Return empty DataFrame
        
        return pd.read_parquet(path)
    
    except Exception as e:
        st.warning(f"Failed to load phytochemical metadata: {e}")
        return pd.DataFrame()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def nearest_neighbors(library_matrix, query_vecs, topk=5):
    """
    Find nearest neighbors in feature space using cosine distance.
    
    Args:
        library_matrix: [n_library, dim] - reference library vectors
        query_vecs: [n_query, dim] - query vectors
        topk: number of nearest neighbors to return
    
    Returns:
        List of lists containing indices of nearest neighbors for each query
    """
    # Ensure inputs are 2D
    if query_vecs.ndim == 1:
        query_vecs = query_vecs.reshape(1, -1)
    
    if library_matrix.shape[0] == 0:
        return [[] for _ in range(query_vecs.shape[0])]
    
    # Compute cosine distances
    dists = cosine_distances(query_vecs, library_matrix)
    
    # Get top-k nearest indices for each query
    topk = min(topk, library_matrix.shape[0])
    nn_indices = []
    
    for i in range(dists.shape[0]):
        sorted_idx = np.argsort(dists[i])[:topk]
        nn_indices.append(sorted_idx.tolist())
    
    return nn_indices


@st.cache_data
def get_ncbi_email():
    """
    Get NCBI email from Streamlit secrets or use default.
    
    For production deployment, set this in Streamlit Cloud secrets:
    Settings > Secrets > Add:
    ```
    [ncbi]
    email = "your.email@institution.edu"
    ```
    """
    try:
        return st.secrets["ncbi"]["email"]
    except:
        # Fallback to environment variable
        email = os.getenv("NCBI_EMAIL", "example@example.com")
        
        if email == "example@example.com":
            st.warning(
                "⚠️ Using default NCBI email. "
                "Please configure in Streamlit Secrets for production."
            )
        
        return email


# ============================================================================
# STARTUP VALIDATION
# ============================================================================

def validate_system_ready():
    """
    Run comprehensive system checks on startup.
    Call this in streamlit_app.py before allowing user interaction.
    """
    checks = {
        "Drug Featurizer": lambda: load_drug_featurizer() is not None,
        "Protein Featurizer": lambda: load_protein_featurizer() is not None,
        "Association Model": lambda: load_association_model() is not None,
        "C-VAE Model": lambda: load_cvae_model() is not None,
        "ID Maps": lambda: load_id_maps() is not None,
        "PharmGKB Library": lambda: load_pharmgkb_drug_library() is not None,
    }
    
    results = {}
    for name, check_fn in checks.items():
        try:
            check_fn()
            results[name] = "✅"
        except:
            results[name] = "❌"
    
    return results
