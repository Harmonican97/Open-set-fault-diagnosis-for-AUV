import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedMultiScaleEncoder(nn.Module):
    """
    Multi-branch dilated 1D-CNN encoder.
    Input:  x of shape (B, C, L)
    Output: z of shape (B, 128)
    """
    def __init__(self, input_dim: int = 16):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(input_dim, 32, 7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )
        self.p3 = nn.Sequential(
            nn.Conv1d(input_dim, 32, 7, padding=6, dilation=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(96, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.p1(x), self.p2(x), self.p3(x)], dim=1)
        return self.fusion(feat).squeeze(-1)


class ProtoNetMDAG(nn.Module):
    """
    Final model (physics removed):
      - Prototypical metric learning for known-class classification
      - Class-conditional Mahalanobis score for open-set rejection (MDAG)

    Closely matches your original PhysProtoNet interface.
    """
    def __init__(self, input_dim: int = 16, feature_dim: int = 128, num_known_classes: int = 3):
        super().__init__()
        self.encoder = DilatedMultiScaleEncoder(input_dim)
        self.prototypes = nn.Parameter(torch.randn(num_known_classes, feature_dim))
        self.register_buffer("running_var", torch.ones(num_known_classes, feature_dim))

    def forward(self, x: torch.Tensor):
        """
        Returns:
          z:         (B, d)
          distances: (B, K) squared Euclidean distances to prototypes
        """
        z = self.encoder(x)
        distances = torch.cdist(z, self.prototypes, p=2) ** 2
        return z, distances

    def compute_v15_loss(self, z: torch.Tensor, dists: torch.Tensor, labels: torch.Tensor, margin: float = 20.0) -> torch.Tensor:
        """
        Same as your original implementation.
        """
        batch_size = labels.size(0)
        pos_dist = dists[torch.arange(batch_size, device=labels.device), labels]

        mask = torch.ones_like(dists).scatter_(1, labels.unsqueeze(1), 0.0)
        neg_dist, _ = torch.min(dists + (1 - mask) * 1e6, dim=1)

        loss_margin = torch.relu(pos_dist - neg_dist + margin).mean()
        loss_var = F.mse_loss(z, self.prototypes[labels])
        return loss_margin + 0.8 * loss_var

    @torch.no_grad()
    def mahalanobis_score(self, z: torch.Tensor, pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Class-conditional diagonal Mahalanobis score, using predicted class 'pred'.
        """
        var = self.running_var[pred] + eps
        diff = z - self.prototypes[pred]
        return torch.sum((diff ** 2) / var, dim=1)

    @torch.no_grad()
    def forward_with_scores(self, x: torch.Tensor, eps: float = 1e-12):
        """
        Returns:
          z, dists, pred, s_md
        """
        z, dists = self.forward(x)
        _, pred = torch.min(dists, dim=1)
        s_md = self.mahalanobis_score(z, pred, eps=eps)
        return z, dists, pred, s_md

    @torch.no_grad()
    def predict_open_set(self, x: torch.Tensor, tau_md: float, unknown_label: int = -1, eps: float = 1e-12) -> torch.Tensor:
        """
        Open-set decision:
          if s_MD >= tau_md -> unknown_label
          else -> predicted known class
        """
        _, _, pred, s_md = self.forward_with_scores(x, eps=eps)
        out = pred.clone()
        out[s_md >= tau_md] = unknown_label
        return out

    @torch.no_grad()
    def update_running_var(self, loader, device=None, eps: float = 1e-5):
        """
        Re-estimates running_var from embeddings on a (known) dataset.
        Recommended to call after training.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_z, all_l = [], []
        for inputs, labels in loader:
            inputs = inputs.to(device)
            z, _ = self.forward(inputs)
            all_z.append(z.cpu())
            all_l.append(labels.cpu())

        all_z = torch.cat(all_z, dim=0)
        all_l = torch.cat(all_l, dim=0)

        num_classes = self.prototypes.size(0)
        for c in range(num_classes):
            z_c = all_z[all_l == c]
            if z_c.size(0) > 1:
                self.running_var[c] = torch.var(z_c, dim=0) + eps


@torch.no_grad()
def calibrate_md_threshold(model: ProtoNetMDAG, calib_loader, quantile: float = 0.95, device=None, eps: float = 1e-12) -> float:
    """
    Quantile-based calibration for open-set rejection:
      tau_md = Quantile_q( s_MD(x) ) over a calibration set of KNOWN samples.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    scores = []
    for inputs, _labels in calib_loader:
        inputs = inputs.to(device)
        _, _, _pred, s_md = model.forward_with_scores(inputs, eps=eps)
        scores.append(s_md.detach().cpu())

    scores = torch.cat(scores, dim=0).numpy()
    import numpy as np
    return float(np.quantile(scores, quantile))


def compute_topology_loss(prototypes: torch.Tensor, similarity_matrix: torch.Tensor) -> torch.Tensor:
    dist_proto = torch.cdist(prototypes, prototypes, p=2)
    dist_proto = dist_proto / (dist_proto.max() + 1e-6)
    target_dist = 1.0 - similarity_matrix
    return F.mse_loss(dist_proto, target_dist)


def get_haizhe_similarity_matrix(num_classes: int = 3) -> torch.Tensor:
    if num_classes == 3:
        return torch.tensor(
            [
                [1.0, 0.1, 0.1],
                [0.1, 1.0, 0.05],
                [0.1, 0.05, 1.0],
            ]
        )
    return torch.eye(num_classes)
