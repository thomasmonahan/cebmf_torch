import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.posterior import posterior_mean_norm


# -------------------------
# Dataset
# -------------------------
class DensityRegressionDataset(Dataset):
    def __init__(self, X, betahat, sebetahat):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.betahat = torch.as_tensor(betahat, dtype=torch.float32)
        self.sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]


# -------------------------
# Mixture Density Network
# -------------------------

def knn_graph_torch(coords, k):
    """
    Build a kNN graph from coordinates tensor [N, d].
    Returns edge_index [2, E].
    """
    N = coords.size(0)
    if N < 2:
        # no edges possible
        return torch.empty((2, 0), dtype=torch.long, device=coords.device)

    k_eff = min(k, N - 1)
    dists = torch.cdist(coords, coords)  # [N, N]
    knn_idx = dists.topk(k_eff + 1, largest=False).indices[:, 1:]  # exclude self
    row = torch.arange(N, device=coords.device).repeat_interleave(k_eff)
    col = knn_idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


# -------------------------
# GNN-based Mixture Density Network
# -------------------------
class GraphMDNSEPARABLE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_gaussians, k=3, n_layers=2, location_dim=2, link_across=False):
        super().__init__()
        self.k = k
        self.location_dim = location_dim
        self.link_across = link_across

        # Exclude (x,y,slice_id) from input features
        feature_dim = input_dim - (location_dim + 1)

        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(feature_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # MDN heads
        self.pi = nn.Linear(hidden_dim, n_gaussians)
        self.mu = nn.Linear(hidden_dim, n_gaussians)
        self.log_sigma = nn.Linear(hidden_dim, n_gaussians)

    def forward(self, x):
        """
        Args:
            x: tensor of shape [N, input_dim], where:
               - first (input_dim - 3) = features
               - next 2 = (x, y)
               - last = slice_id
        """
        feats = x[:, :-1-self.location_dim]       # embeddings/features
        coords = x[:, -1-self.location_dim:-1]    # x,y
        slice_ids = x[:, -1].long()

        edge_indices = []
        # Build separate kNN graphs per slice
        for slice_id in slice_ids.unique():
            mask = slice_ids == slice_id
            coords_slice = coords[mask]
            if coords_slice.size(0) > 1:  # need at least 2 nodes
                edge_index = knn_graph_torch(coords_slice, k=self.k)
                # remap to global indices
                global_idx = mask.nonzero(as_tuple=False).view(-1)
                edge_index = global_idx[edge_index]
                edge_indices.append(edge_index)

        # Optionally link across slices (full bipartite or simple concat)
        if self.link_across and slice_ids.unique().numel() > 1:
            idx_a = (slice_ids == slice_ids.unique()[0]).nonzero(as_tuple=False).view(-1)
            idx_b = (slice_ids == slice_ids.unique()[1]).nonzero(as_tuple=False).view(-1)
            # fully connect A <-> B
            cross_edges = torch.cartesian_prod(idx_a, idx_b).T
            edge_indices.append(cross_edges)

        # Combine edge indices
        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            # Fallback: no edges
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        # Run GNN
        h = feats
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))

        # MDN heads
        pi = torch.softmax(self.pi(h), dim=-1)
        mu = self.mu(h)
        log_sigma = torch.clamp(self.log_sigma(h), -10, 5)

        return pi, mu, log_sigma

# -------------------------
# Loss function
# -------------------------
def mdn_loss_with_varying_noise(pi, mu, log_sigma, betahat, sebetahat):
    #sigma = torch.exp(log_sigma)
    sigma = torch.exp(log_sigma)  # prevent too small or too large
    #sigma = 0.01 + 0.99 * torch.sigmoid(log_sigma)

    total_sigma = torch.sqrt(sigma**2 + sebetahat.unsqueeze(1) ** 2)
    dist = torch.distributions.Normal(mu, total_sigma)
    log_probs = dist.log_prob(betahat.unsqueeze(1)) + torch.log(pi)
    return -torch.logsumexp(log_probs, dim=1).mean()

import warnings

def validate_sebetahat(sebetahat: torch.Tensor, min_val: float = 1e-8) -> torch.Tensor:
    """Validate sebetahat:
    - Raise error if NaNs are found
    - Clamp zeros/negatives to min_val and warn
    """
    if torch.isnan(sebetahat).any():
        raise ValueError("NaN detected in sebetahat! Please clean your input.")

    if (sebetahat <= 0).any():
        warnings.warn(
            f"Non-positive values detected in sebetahat. "
            f"Clamping to {min_val} to avoid numerical issues.",
            RuntimeWarning
        )
        sebetahat = torch.clamp(sebetahat, min=min_val)

    return sebetahat


# -------------------------
# Result container
# -------------------------
class EmdnPosteriorMeanNorm:
    def __init__(
        self,
        post_mean,
        post_mean2,
        post_sd,
        location,
        pi_np,
        scale,
        loss=0,
        model_param=None,
    ):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.location = location
        self.pi_np = pi_np
        self.scale = scale
        self.loss = loss
        self.model_param = model_param


# -------------------------
# Main solver
# -------------------------
def egnnmdnseparable_posterior_means(
    X,
    betahat,
    sebetahat,
    n_epochs=50,
    n_layers=4,
    n_gaussians=5,
    hidden_dim=64,
    batch_size=512,
    lr=1e-4,
    model_param=None,
):
    # Standardize X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dataset + DataLoader
    sebetahat = validate_sebetahat(torch.as_tensor(sebetahat, dtype=torch.float32))

    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Init model
    model = GraphMDNSEPARABLE(
        input_dim=X_scaled.shape[1],
        hidden_dim=hidden_dim,
        n_gaussians=n_gaussians,
        n_layers=n_layers,
    )
    if model_param is not None:
        model.load_state_dict(model_param)
    rstrength = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=rstrength)
    print('reg strength:', rstrength)
    # Training loop
    losses = []
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets, noise_std in dataloader:
            optimizer.zero_grad()
            pi, mu, log_sigma = model(inputs)
            loss = mdn_loss_with_varying_noise(pi, mu, log_sigma, targets, noise_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            for name, param in model.named_parameters():
            	if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")

            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[EGNNMDNSEPARABLE] Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(dataloader):.4f}")
        losses.append(running_loss / len(dataloader))
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()

    # Prediction for all data
    model.eval()
    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for X_batch, _, _ in full_loader:
            pi, mu, log_sigma = model(X_batch)

    # Posterior means per observation
    J = len(betahat)
    post_mean = torch.empty(J, dtype=torch.float32)
    post_mean2 = torch.empty(J, dtype=torch.float32)
    post_sd = torch.empty(J, dtype=torch.float32)
    for i in range(len(betahat)):
        data_loglik = get_data_loglik_normal_torch(
            betahat=betahat[i : (i + 1)],
            sebetahat=sebetahat[i : (i + 1)],
            location=mu[i, :],
            scale=torch.exp(log_sigma)[i, :],
        )
        result = posterior_mean_norm(
            betahat=betahat[i : (i + 1)],
            sebetahat=sebetahat[i : (i + 1)],
            log_pi=torch.log(pi[i, :]),
            data_loglik=data_loglik,
            location=mu[i, :],
            scale=torch.exp(log_sigma)[i, :],
        )
        post_mean[i] = result.post_mean
        post_mean2[i] = result.post_mean2
        post_sd[i] = result.post_sd

    return EmdnPosteriorMeanNorm(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        location=mu,
        pi_np=pi,
        scale=torch.exp(log_sigma),
        loss=running_loss,
        model_param=model.state_dict(),
    )
