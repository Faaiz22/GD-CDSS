
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import itertools

# 20 standard amino acids
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_LIST = list(AA)
AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

# Pre-compute all possible dipeptides (400)
DIPEPTIDES = [a + b for a in AA_LIST for b in AA_LIST]
DPC_INDEX = {d: i for i, d in enumerate(DIPEPTIDES)}


# ------------------------------- Utility --------------------------------

def compute_dpc(seq: str):
    """
    Compute 400-dimensional Dipeptide Composition vector.
    """
    seq = seq.upper()
    vec = np.zeros(400, dtype=float)
    total = len(seq) - 1
    if total <= 0:
        return vec

    for i in range(total):
        di = seq[i:i+2]
        if di in DPC_INDEX:
            vec[DPC_INDEX[di]] += 1.0
    vec /= total
    return vec


def compute_physicochemical_features(seq: str):
    """
    Compute 7 physicochemical sequence-level features.
    """
    seq = seq.upper()
    if not seq:
        return np.zeros(7, dtype=float)

    mw_map = {
        'A':89.1,'C':121.2,'D':133.1,'E':147.1,'F':165.2,'G':75.1,'H':155.2,'I':131.2,
        'K':146.2,'L':131.2,'M':149.2,'N':132.1,'P':115.1,'Q':146.2,'R':174.2,'S':105.1,
        'T':119.1,'V':117.1,'W':204.2,'Y':181.2
    }
    hydrop_map = {
        'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,
        'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,
        'W':-0.9,'Y':-1.3
    }
    pos_set = {'K','R','H'}
    neg_set = {'D','E'}

    mw = np.mean([mw_map.get(a,0) for a in seq])
    hydrop = np.mean([hydrop_map.get(a,0) for a in seq])
    aromatic = sum(1 for a in seq if a in {'F','W','Y'}) / len(seq)
    charge = (sum(1 for a in seq if a in pos_set) - sum(1 for a in seq if a in neg_set)) / len(seq)
    instability = np.std([mw_map.get(a,0) for a in seq])
    flexibility = np.std([hydrop_map.get(a,0) for a in seq])
    length = len(seq) / 1000.0

    return np.array([mw, hydrop, aromatic, charge, instability, flexibility, length], dtype=float)


# ---------------------------- Main Featurizer ----------------------------

class ProteinFeaturizer:
    """
    38-dim protein vector:
      [ 31 PCA DPC components | 7 scaled physchem features ]
    """

    def __init__(self, pca_model=None, scaler_physchem=None, scaler_dpc=None):
        self.pca_model = pca_model
        self.scaler_physchem = scaler_physchem
        self.scaler_dpc = scaler_dpc

    # ------------------- Fitting -------------------

    def fit_dpc_pca(self, seq_list, n_components=31):
        """
        Fit scaler + PCA for 400-d DPC vectors.
        """
        dpcs = []
        for seq in seq_list:
            dpcs.append(compute_dpc(seq))
        dpcs = np.vstack(dpcs)

        self.scaler_dpc = StandardScaler().fit(dpcs)
        dpcs_scaled = self.scaler_dpc.transform(dpcs)
        self.pca_model = PCA(n_components=n_components).fit(dpcs_scaled)
        return self

    def fit_physchem_scaler(self, seq_list):
        """
        Fit scaler for 7 physicochemical features.
        """
        feats = []
        for seq in seq_list:
            feats.append(compute_physicochemical_features(seq))
        feats = np.vstack(feats)
        self.scaler_physchem = StandardScaler().fit(feats)
        return self

    # ------------------- Feature Construction -------------------

    def build_feature_vector(self, seq: str):
        """
        Convert raw amino acid sequence → 38-dim vector.
        """
        seq = seq.upper()

        dpc400 = compute_dpc(seq).reshape(1, -1)
        dpc_scaled = self.scaler_dpc.transform(dpc400)
        dpc31 = self.pca_model.transform(dpc_scaled).flatten()

        phys7 = compute_physicochemical_features(seq).reshape(1, -1)
        phys7_scaled = self.scaler_physchem.transform(phys7).flatten()

        return np.concatenate([dpc31, phys7_scaled], axis=0)

    # ------------------- Persistence -------------------

    def save(self, path_prefix):
        joblib.dump(self.scaler_physchem, f"{path_prefix}_physchem_scaler.pkl")
        joblib.dump(self.scaler_dpc, f"{path_prefix}_dpc_scaler.pkl")
        joblib.dump(self.pca_model, f"{path_prefix}_dpc_pca.pkl")

    def load(self, path_prefix):
        self.scaler_physchem = joblib.load(f"{path_prefix}_physchem_scaler.pkl")
        self.scaler_dpc = joblib.load(f"{path_prefix}_dpc_scaler.pkl")
        self.pca_model = joblib.load(f"{path_prefix}_dpc_pca.pkl")
        return self
