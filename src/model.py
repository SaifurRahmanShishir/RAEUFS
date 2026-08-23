"""RAEUFS model components.

This module preserves the current project training design:
- W, A, encoder, and decoder are optimized batch-by-batch.
- F and S are updated at the end of each epoch.
- W is kept orthogonal 
"""

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class FeatureSelector(nn.Module):
    def __init__(self, input_dim: int, selected_features: int):
        super().__init__()
        self.W = nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(input_dim, selected_features))
        )

    def forward(self, x):
        x = x.to(self.W.device)
        return self.W.T @ x.T


class RSRLayer(nn.Module):
    def __init__(self, d_A: int, d_bar: int):
        super().__init__()
        self.d_A = d_A
        self.d_bar = d_bar
        self.A = nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(d_A, d_bar))
        )

    def forward(self, z):
        z_hat = self.A @ z.view(z.size(0), self.d_bar, 1)
        return z_hat.squeeze(2)


class RSRAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        selected_features,
        d_A,
        d_bar,
        batch_size,
        lr,
        lambda1,
        lambda2,
        alpha,
        beta,
        gamma,
        theta,
        sigma,
        train_data,
        val_data=None,
        update_S_every=1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_data", "val_data"])
        self.automatic_optimization = False

        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.sigma = sigma
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.update_S_every = update_S_every

        self.epoch_losses = {
            "total_loss_W_A": [],
            "reconstruction_loss": [],
            "rsr_loss": [],
            "sparsity_loss": [],
        }
        self.train_losses = []

        self.FS = FeatureSelector(input_dim, selected_features)

        self.encoder = nn.Sequential(
            nn.Linear(selected_features, selected_features // 2),
            nn.BatchNorm1d(selected_features // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(selected_features // 2, selected_features // 4),
            nn.BatchNorm1d(selected_features // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(selected_features // 4, selected_features),
            nn.BatchNorm1d(selected_features),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(selected_features, d_bar),
            nn.Sigmoid(),
        )

        self.rsr = RSRLayer(d_A + 1, d_bar)

        self.decoder = nn.Sequential(
            nn.Linear(d_A + 1, d_bar),
            nn.BatchNorm1d(d_bar),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(d_bar, selected_features // 8),
            nn.BatchNorm1d(selected_features // 8),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(selected_features // 8, selected_features // 4),
            nn.BatchNorm1d(selected_features // 4),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(selected_features // 4, selected_features // 2),
            nn.BatchNorm1d(selected_features // 2),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(selected_features // 2, selected_features),
            nn.BatchNorm1d(selected_features),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(selected_features, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.to(self.device)
        enc = self.encoder(self.FS(x).T)
        latent = self.rsr(enc)
        dec = self.decoder(latent)
        return enc, dec, latent, self.rsr.A, self.FS.W

    def training_step(self, batch, batch_idx):
        x, _ = batch
        opt_W_A = self.optimizers()

        # Batch-by-batch update of W, A, encoder, and decoder.
        opt_W_A.zero_grad()
        enc, dec, _, A, W = self(x)

        reconstruction_loss = torch.norm(x - dec, p=2, dim=1).sum()

        # A has shape (d_A + 1, d_bar) and represents the transpose of the
        # paper's A under the original implementation convention.
        M = A.T @ A
        rsr_recon = torch.norm(enc - enc @ M, p=2, dim=1).sum()
        rsr_loss = (
            self.lambda1 * rsr_recon
            + self.lambda2
            * torch.norm(
                A @ A.T - torch.eye(A.size(0), device=self.device), p=2
            )
            ** 2
        )

        sparsity_loss = self.alpha * torch.sum(torch.norm(W, p=2, dim=1))
        total_loss_W_A = reconstruction_loss + rsr_loss + sparsity_loss

        self.manual_backward(total_loss_W_A)

        with torch.no_grad():
            self.FS.W.copy_(self.orthogonal_projection(self.FS.W))

        opt_W_A.step()

        self.epoch_losses["total_loss_W_A"].append(total_loss_W_A.detach())
        self.epoch_losses["reconstruction_loss"].append(reconstruction_loss.detach())
        self.epoch_losses["rsr_loss"].append(rsr_loss.detach())
        self.epoch_losses["sparsity_loss"].append(sparsity_loss.detach())
        self.train_losses.append(total_loss_W_A.detach().cpu().item())

        return total_loss_W_A

    def on_train_epoch_end(self):
        """Update F and S after the neural-network batch updates for the epoch."""
        if self.current_epoch % self.update_S_every != 0:
            return

        with torch.no_grad():
            # Fixed sample ordering is important because F is preserved across epochs. 
            full_x = self.train_data.X.to(self.device)

            enc_full = self.encoder(self.FS(full_x).T)
            projected = enc_full @ self.rsr.A.T

            S = self.solve_S(projected, self.beta)
            L_s = self.compute_graph_laplacian(S)

            if not hasattr(self, "F") or self.F is None:
                self.F = None

            self.F = self.solve_F(
                projected,
                S,
                self.gamma,
                num_iters=100,
                F=self.F,
            )

            cluster_loss = self.theta * torch.norm(
                projected - self.F, p="fro"
            ) ** 2
            graph_loss = self.gamma * torch.trace(self.F.T @ L_s @ self.F)
            entropy_penalty = self.gamma * self.beta * torch.sum(
                S * torch.log(S + 1e-12)
            )

        mean_total_loss_W_A = torch.stack(
            self.epoch_losses["total_loss_W_A"]
        ).mean()
        mean_reconstruction_loss = torch.stack(
            self.epoch_losses["reconstruction_loss"]
        ).mean()
        mean_rsr_loss = torch.stack(self.epoch_losses["rsr_loss"]).mean()
        mean_sparsity_loss = torch.stack(
            self.epoch_losses["sparsity_loss"]
        ).mean()

        self.epoch_losses = {
            "total_loss_W_A": [],
            "reconstruction_loss": [],
            "rsr_loss": [],
            "sparsity_loss": [],
        }

        self.log(
            "total_loss",
            mean_total_loss_W_A + cluster_loss + graph_loss + entropy_penalty,
            prog_bar=True,
        )
        self.log("reconstruction_loss", mean_reconstruction_loss, prog_bar=True)
        self.log("rsr_loss", mean_rsr_loss, prog_bar=True)
        self.log("sparsity_loss", mean_sparsity_loss, prog_bar=True)
        self.log("cluster_loss", cluster_loss, prog_bar=True)
        self.log("graph_loss", graph_loss, prog_bar=True)
        self.log("entropy_penalty", entropy_penalty, prog_bar=True)

    def configure_optimizers(self):
        opt_W_A = torch.optim.AdamW(
            [
                {"params": self.FS.W},
                {"params": self.rsr.A},
                {"params": self.encoder.parameters()},
                {"params": self.decoder.parameters()},
            ],
            lr=self.lr,
            weight_decay=1e-4,
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler_W_A = torch.optim.lr_scheduler.OneCycleLR(
            opt_W_A,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=30,
            final_div_factor=100,
        )

        return [opt_W_A], [{"scheduler": scheduler_W_A, "interval": "step"}]

    @staticmethod
    def solve_F(Z, S, gamma, num_iters=50, F=None):
        """Solve for F with the GPI implementation."""
        n_samples = Z.shape[0]
        num_classes = Z.shape[1]

        I = torch.eye(n_samples, device=Z.device)
        ones = torch.ones((n_samples, n_samples), device=Z.device) / n_samples
        H = I - ones
        P = torch.diag(S.sum(dim=1))
        Ls = P - (S + S.T) / 2
        A = H + 2 * gamma * Ls
        C = H @ Z

        nu = torch.max(torch.linalg.eigvalsh(A.detach().cpu())).item() + 1e-5
        A_tilde = nu * I - A

        if F is None:
            F_cpu = torch.randn((n_samples, num_classes), dtype=Z.dtype)
            F_cpu = torch.nn.init.orthogonal_(F_cpu)
            F = F_cpu.to(Z.device)

        for _ in range(num_iters):
            R = 2 * A_tilde @ F + 2 * C
            R_cpu = R.detach().cpu()
            U, _, Vh = torch.linalg.svd(R_cpu, full_matrices=False)
            F = (U @ Vh).to(device=Z.device, dtype=Z.dtype)

        return F

    @staticmethod
    def solve_S(Z, beta):
        """Current closed-form similarity update """
        dist_matrix = torch.cdist(Z, Z, p=2) ** 2
        S = torch.exp(-dist_matrix / (2 * beta))
        S = S / (S.sum(dim=1, keepdim=True) + 1e-8)
        S = (S + S.T) / 2
        return S

    @staticmethod
    def compute_similarity_matrix(X, sigma=1.0):
        pairwise_diff = X.unsqueeze(1) - X.unsqueeze(0)
        pairwise_dist = torch.norm(pairwise_diff, dim=2, p=2)
        S = torch.exp(-(pairwise_dist**2) / (2 * sigma**2))
        S = (S + S.T) / 2
        S = S / S.sum(dim=1, keepdim=True)
        return S

    @staticmethod
    def compute_graph_laplacian(S):
        P = torch.diag(S.sum(dim=1))
        return P - S

    @staticmethod
    def orthogonal_projection(matrix):
        matrix_cpu = matrix.detach().cpu()
        try:
            U, _, Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
        except RuntimeError:
            matrix_cpu = matrix_cpu + torch.randn_like(matrix_cpu) * 1e-7
            U, _, Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
        return (U @ Vh).to(device=matrix.device, dtype=matrix.dtype)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    def val_dataloader(self):
        if self.val_data is None:
            return None
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
