import numpy as np
import torch
import torch.nn as nn


def _inv_sqrt_sym(cov: torch.Tensor, epsilon: float = 1e-5, ret_sqrt: bool = True) -> torch.Tensor:
    """Symmetric eigendecomposition inverse / inverse square-root in float64."""
    cov64 = 0.5 * (cov + cov.transpose(-1, -2)).to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(cov64)
    mask = eigvals > epsilon
    safe_vals = torch.where(mask, eigvals, torch.ones_like(eigvals))
    inv_vals = torch.where(
        mask,
        torch.rsqrt(safe_vals) if ret_sqrt else (1.0 / safe_vals),
        torch.zeros_like(eigvals),
    )
    out = (eigvecs * inv_vals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
    return out.to(cov.dtype)


def vamp2_score(
    chi_0: torch.Tensor,
    chi_tau: torch.Tensor,
    epsilon: float = 1e-5,
    k_eig: int = 0,
    include_constant: bool = False,
) -> torch.Tensor:
    """
    Compute VAMP-2 score following txie-93/gdynet/gdynet/vampnet.py.

    chi_0, chi_tau: (N_samples, n_classes)
    Returns mean-free VAMP-2 score in [0, n_classes - 1] (or [1, n_classes] if include_constant=True).
    """
    x = chi_0.reshape(-1, chi_0.shape[-1]).to(torch.float64)
    y = chi_tau.reshape(-1, chi_tau.shape[-1]).to(torch.float64)
    b = float(x.shape[0])
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    xm = (x - x_mean).t()
    ym = (y - y_mean).t()

    denom = max(b - 1.0, 1.0)
    cov_01 = (xm @ ym.t()) / denom
    cov_00 = (xm @ xm.t()) / denom
    cov_11 = (ym @ ym.t()) / denom

    cov_00_ir = _inv_sqrt_sym(cov_00, epsilon=epsilon, ret_sqrt=True)
    cov_11_ir = _inv_sqrt_sym(cov_11, epsilon=epsilon, ret_sqrt=True)
    vamp_matrix = cov_00_ir @ cov_01 @ cov_11_ir

    if k_eig > 0:
        svals = torch.linalg.svdvals(vamp_matrix)
        score = torch.sum(svals[:k_eig] ** 2)
    else:
        score = torch.sum(vamp_matrix ** 2)

    if include_constant:
        score = score + 1.0
    return score


class VampLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-5, k_eig: int = 0):
        super().__init__()
        self.epsilon = epsilon
        self.k_eig = k_eig

    def forward(self, chi_0: torch.Tensor, chi_tau: torch.Tensor) -> torch.Tensor:
        return -vamp2_score(chi_0, chi_tau, epsilon=self.epsilon, k_eig=self.k_eig, include_constant=False)


def estimate_koopman_op(traj: np.ndarray, tau: int, epsilon: float = 1e-5) -> np.ndarray:
    """
    Estimate Koopman operator K = C00^{-1} C0tau on trajectory array (F, n_targets, n_classes).
    Matches txie-93/gdynet/gdynet/vampnet.py::estimate_koopman_op.
    """
    n_classes = traj.shape[-1]
    prev = traj[:-tau].reshape(-1, n_classes).astype(np.float64)
    post = traj[tau:].reshape(-1, n_classes).astype(np.float64)

    c_0 = prev.T @ prev
    c_tau = prev.T @ post

    eigv, eigvec = np.linalg.eigh(0.5 * (c_0 + c_0.T))
    include = eigv > epsilon
    eigv = eigv[include]
    eigvec = eigvec[:, include]
    c0_inv = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    return c0_inv @ c_tau


def compute_its(traj: np.ndarray, lags: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    n_classes = traj.shape[-1]
    its = np.zeros((n_classes - 1, len(lags)), dtype=np.float64)
    for idx, tau_lag in enumerate(lags):
        k_op = estimate_koopman_op(traj, int(tau_lag), epsilon=epsilon)
        k_eigvals, _ = np.linalg.eig(np.real(k_op))
        k_eigvals = np.sort(np.abs(k_eigvals))[:-1]
        k_eigvals = np.clip(k_eigvals, 1e-12, 1.0 - 1e-12)
        its[:, idx] = -float(tau_lag) / np.log(k_eigvals)
    return its


def compute_ck_test(traj: np.ndarray, steps: int, tau: int, epsilon: float = 1e-5):
    """Returns (predicted, estimated) each with shape (n_states, n_states, steps) and CK RMSE."""
    n_states = traj.shape[-1]
    predicted = np.zeros((n_states, n_states, steps), dtype=np.float64)
    estimated = np.zeros((n_states, n_states, steps), dtype=np.float64)
    predicted[:, :, 0] = np.eye(n_states)
    estimated[:, :, 0] = np.eye(n_states)

    koop_1 = estimate_koopman_op(traj, tau, epsilon=epsilon)
    for n in range(1, steps):
        koop_pred = np.linalg.matrix_power(koop_1, n)
        koop_est = estimate_koopman_op(traj, tau * n, epsilon=epsilon)
        predicted[:, :, n] = koop_pred
        estimated[:, :, n] = koop_est

    ck_rmse = float(np.sqrt(np.mean((predicted[:, :, 1:] - estimated[:, :, 1:]) ** 2)))
    return predicted, estimated, ck_rmse
