import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MASS_EFF = 6.5737 
B_MINUS_G = 3.86
DRAG_COEFFS = [0.3747, 36.35]

class DilatedMultiScaleEncoder(nn.Module):
    def __init__(self, input_dim=16):
        super(DilatedMultiScaleEncoder, self).__init__()
        self.p1 = nn.Sequential(nn.Conv1d(input_dim, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv1d(input_dim, 32, 7, padding=3), nn.GroupNorm(4, 32), nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv1d(input_dim, 32, 7, padding=6, dilation=2), nn.GroupNorm(4, 32), nn.ReLU())
        
        self.fusion = nn.Sequential(
            nn.Conv1d(96, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        feat = torch.cat([self.p1(x), self.p2(x), self.p3(x)], dim=1)
        return self.fusion(feat).squeeze(-1)

class PhysProtoNet(nn.Module):
    def __init__(self, input_dim=16, feature_dim=128, num_known_classes=3):
        super(PhysProtoNet, self).__init__()
        self.encoder = DilatedMultiScaleEncoder(input_dim)
        self.prototypes = nn.Parameter(torch.randn(num_known_classes, feature_dim))
        self.register_buffer("running_var", torch.ones(num_known_classes, feature_dim))

    def forward(self, x, phys_res=None):
        z = self.encoder(x)
        distances = torch.cdist(z, self.prototypes, p=2)**2
        return z, distances

    def compute_v15_loss(self, z, dists, labels, margin=20.0):
        batch_size = labels.size(0)
        pos_dist = dists[torch.arange(batch_size), labels]
        mask = torch.ones_like(dists).scatter_(1, labels.unsqueeze(1), 0.)
        neg_dist, _ = torch.min(dists + (1 - mask) * 1e6, dim=1)
        loss_margin = torch.relu(pos_dist - neg_dist + margin).mean()
        
        loss_var = F.mse_loss(z, self.prototypes[labels])
        
        return loss_margin + 0.8 * loss_var

class HaizhePhysicsEngine:
    @staticmethod
    def get_batch_residuals(inputs):
        pwms, volts = inputs[:, 0:4, -1], inputs[:, 7, -1]
        depth_seq, acc_z_meas = inputs[:, 5, :], inputs[:, 12, -1]
        if volts.dim() == 1: volts = volts.unsqueeze(1)
        omega_sq = (pwms - 1000.0).clamp(min=0) * (volts / 12.0) * 14.5
        f_z = -torch.sum(3.065e-4 * omega_sq, dim=1)
        w = (depth_seq[:, -1] - depth_seq[:, -2]) / 0.1
        drag = DRAG_COEFFS[0] * w + DRAG_COEFFS[1] * w * torch.abs(w)
        theo_a = (f_z + B_MINUS_G - drag) / MASS_EFF
        res_vec = torch.zeros(inputs.size(0), 16).to(inputs.device)
        res_vec[:, 12] = torch.abs(theo_a - acc_z_meas)
        return res_vec

def compute_topology_loss(prototypes, similarity_matrix):
    dist_proto = torch.cdist(prototypes, prototypes, p=2)
    dist_proto = dist_proto / (dist_proto.max() + 1e-6)
    target_dist = 1.0 - similarity_matrix
    return F.mse_loss(dist_proto, target_dist)

# def get_haizhe_similarity_matrix():
#     return torch.tensor([[1.0, 0.1, 0.1], [0.1, 1.0, 0.05], [0.1, 0.05, 1.0]])

def get_haizhe_similarity_matrix(num_classes=3):
    if num_classes == 3:
        return torch.tensor([[1.0, 0.1, 0.1], 
                             [0.1, 1.0, 0.05], 
                             [0.1, 0.05, 1.0]])
    else:
        return torch.eye(num_classes)