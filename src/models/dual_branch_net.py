
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchNet(nn.Module):
    """
    Dual-branch neural network for predicting Gene–Drug associations.

    Left branch  (Drug vector):    135 → 256 → 128
    Right branch (Protein vector): 38  → 128 → 64
    Final fusion: Concatenate → MLP → Scalar score
    """

    def __init__(self, drug_dim=135, prot_dim=38, hidden_dim=256, dropout=0.25):
        super().__init__()

        # Drug branch
        self.drug_fc1 = nn.Linear(drug_dim, hidden_dim)
        self.drug_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.drug_dp = nn.Dropout(dropout)

        # Protein branch
        self.prot_fc1 = nn.Linear(prot_dim, hidden_dim // 2)
        self.prot_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.prot_dp = nn.Dropout(dropout)

        # Fusion + Output
        fused_dim = (hidden_dim // 2) + (hidden_dim // 4)
        self.out_fc1 = nn.Linear(fused_dim, hidden_dim // 2)
        self.out_fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, drug_vec, prot_vec):
        # Drug branch
        d = F.relu(self.drug_fc1(drug_vec))
        d = self.drug_dp(d)
        d = F.relu(self.drug_fc2(d))

        # Protein branch
        p = F.relu(self.prot_fc1(prot_vec))
        p = self.prot_dp(p)
        p = F.relu(self.prot_fc2(p))

        # Fusion
        x = torch.cat([d, p], dim=1)

        # Output
        x = F.relu(self.out_fc1(x))
        x = torch.sigmoid(self.out_fc2(x))

        return x
