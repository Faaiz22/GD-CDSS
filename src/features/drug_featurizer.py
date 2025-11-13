
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


class DrugFeaturizer:
    """
    Re-engineered 135-dim Drug Feature Builder.
    Vector Layout:
      [ 7 scaled RDKit descriptors | 128 PCA fingerprint components ]
    """

    def __init__(self, pca_model=None, scaler_descriptors=None, scaler_fp=None):
        self.pca_model = pca_model
        self.scaler_descriptors = scaler_descriptors
        self.scaler_fp = scaler_fp

    # ------------------------ Core Helpers ------------------------

    def _compute_descriptors(self, mol):
        """
        Compute 7 classical molecular descriptors.
        """
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),
        ], dtype=float)

    def _compute_morgan_fp(self, mol):
        """
        Compute 2048-bit Morgan fingerprint → numpy array of 0/1.
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((1,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(float)

    # ------------------------ Fitting Functions ------------------------

    def fit_descriptor_scaler(self, smiles_list):
        """
        Fit scaler for 7 descriptors.
        """
        descs = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            descs.append(self._compute_descriptors(mol))

        descs = np.vstack(descs)
        self.scaler_descriptors = StandardScaler().fit(descs)
        return self

    def fit_fp_pca(self, smiles_list, n_components=128):
        """
        Fit PCA + scaler for Morgan fingerprints.
        """
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fps.append(self._compute_morgan_fp(mol))

        fps = np.vstack(fps)
        self.scaler_fp = StandardScaler().fit(fps)

        fps_scaled = self.scaler_fp.transform(fps)
        self.pca_model = PCA(n_components=n_components).fit(fps_scaled)
        return self

    # ------------------------ Main Feature Builder ------------------------

    def build_feature_vector(self, smiles: str):
        """
        Convert SMILES → 135-dim feature vector.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        desc7 = self._compute_descriptors(mol).reshape(1, -1)
        desc7_scaled = self.scaler_descriptors.transform(desc7).flatten()

        fp2048 = self._compute_morgan_fp(mol).reshape(1, -1)
        fp2048_scaled = self.scaler_fp.transform(fp2048)
        fp128 = self.pca_model.transform(fp2048_scaled).flatten()

        return np.concatenate([desc7_scaled, fp128], axis=0)

    # ------------------------ Persistence ------------------------

    def save(self, path_prefix):
        joblib.dump(self.scaler_descriptors, f"{path_prefix}_desc_scaler.pkl")
        joblib.dump(self.scaler_fp, f"{path_prefix}_fp_scaler.pkl")
        joblib.dump(self.pca_model, f"{path_prefix}_fp_pca.pkl")

    def load(self, path_prefix):
        self.scaler_descriptors = joblib.load(f"{path_prefix}_desc_scaler.pkl")
        self.scaler_fp = joblib.load(f"{path_prefix}_fp_scaler.pkl")
        self.pca_model = joblib.load(f"{path_prefix}_fp_pca.pkl")
        return self
