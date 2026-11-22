"""Canonical GDyNet and GeoShell-GDyNet architectures with modular ablation support."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion for pairwise interatomic distances."""

    def __init__(self, dmin: float = 0.0, dmax: float = 7.0, step: float = 0.2, var: float = 0.2):
        super().__init__()
        centers = torch.arange(dmin, dmax + step * 0.5, step, dtype=torch.float32)
        self.register_buffer("centers", centers)
        self.inv_var2 = 1.0 / (var ** 2)
        self.out_dim = int(centers.numel())

    def forward(self, dists: torch.Tensor) -> torch.Tensor:
        diff = dists.unsqueeze(-1) - self.centers
        return torch.exp(-(diff ** 2) * self.inv_var2)


class CGCNNLayer(nn.Module):
    """Canonical gated CGCNN convolution layer matching Xie et al. (Nat. Commun. 2019)."""

    def __init__(self, atom_fea_len: int, bond_fea_len: int, use_bn: bool = True):
        super().__init__()
        in_dim = 2 * atom_fea_len + bond_fea_len
        self.fc_core = nn.Linear(in_dim, atom_fea_len)
        self.fc_filter = nn.Linear(in_dim, 1)
        self.use_bn = use_bn
        if use_bn:
            self.bn_core = nn.BatchNorm1d(atom_fea_len)
            self.bn_out = nn.BatchNorm1d(atom_fea_len)

    def forward(self, atom_fea: torch.Tensor, bond_fea: torch.Tensor, nbr_list: torch.Tensor) -> torch.Tensor:
        B, N, H = atom_fea.shape
        M = nbr_list.shape[-1]
        batch_idx = torch.arange(B, device=atom_fea.device)[:, None, None]
        nbr_fea = atom_fea[batch_idx, nbr_list]
        self_fea = atom_fea.unsqueeze(2).expand(-1, -1, M, -1)
        total_fea = torch.cat([self_fea, nbr_fea, bond_fea], dim=-1)

        nbr_core = self.fc_core(total_fea)
        nbr_filter = self.fc_filter(total_fea)
        if self.use_bn:
            nbr_core = self.bn_core(nbr_core.reshape(-1, H)).reshape(B, N, M, H)
        nbr_filter = F.softmax(nbr_filter, dim=2)
        nbr_core = F.relu(nbr_core)
        nbr_sumed = torch.mean(nbr_filter * nbr_core, dim=2)
        if self.use_bn:
            nbr_sumed = self.bn_out(nbr_sumed.reshape(-1, H)).reshape(B, N, H)
        return F.relu(atom_fea + nbr_sumed)


class CanonicalGDyNet(nn.Module):
    """Canonical PyTorch GDyNet baseline (Xie et al., Nat. Commun. 2019)."""

    def __init__(
        self,
        n_classes: int = 4,
        atom_fea_len: int = 16,
        n_conv: int = 3,
        dmin: float = 0.0,
        dmax: float = 7.0,
        step: float = 0.2,
        use_bn: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.atom_fea_len = atom_fea_len
        self.n_conv = n_conv
        self.rbf = GaussianRBF(dmin=dmin, dmax=dmax, step=step, var=step)
        self.embedding = nn.Embedding(100, atom_fea_len)
        self.convs = nn.ModuleList(
            [CGCNNLayer(atom_fea_len, self.rbf.out_dim, use_bn=use_bn) for _ in range(n_conv)]
        )
        self.fc_readout = nn.Linear(atom_fea_len, atom_fea_len)
        self.fc_out = nn.Linear(atom_fea_len, n_classes)

    def forward(
        self,
        atom_types: torch.Tensor,
        nbr_dists: torch.Tensor,
        nbr_lists: torch.Tensor,
        target_index: torch.Tensor,
        nbr_vecs: torch.Tensor = None,
    ) -> torch.Tensor:
        atom_fea = self.embedding(atom_types)
        bond_fea = self.rbf(nbr_dists)
        for conv in self.convs:
            atom_fea = conv(atom_fea, bond_fea, nbr_lists)
        batch_idx = torch.arange(atom_fea.shape[0], device=atom_fea.device)[:, None]
        target_fea = atom_fea[batch_idx, target_index]
        target_fea = F.relu(target_fea)
        target_fea = F.relu(self.fc_readout(target_fea))
        return F.softmax(self.fc_out(target_fea), dim=-1)


class GeoShellGDyNet(nn.Module):
    """GeoShell-GDyNet: Jumping Knowledge, distance-aware 1st-shell attention,
    SO(3)-invariant multipole shell geometry, and calibrated pre-norm residual head.
    """

    def __init__(
        self,
        n_classes: int = 4,
        atom_fea_len: int = 16,
        n_conv: int = 3,
        dmin: float = 0.0,
        dmax: float = 7.0,
        step: float = 0.2,
        use_bn: bool = True,
        width_mult: int = 2,
        use_jk: bool = True,
        use_shell_attn: bool = True,
        use_so3_geo: bool = True,
        use_res_temp: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.atom_fea_len = atom_fea_len
        self.n_conv = n_conv
        self.hidden = width_mult * atom_fea_len
        self.use_jk = use_jk
        self.use_shell_attn = use_shell_attn
        self.use_so3_geo = use_so3_geo
        self.use_res_temp = use_res_temp

        self.rbf = GaussianRBF(dmin=dmin, dmax=dmax, step=step, var=step)
        self.embedding = nn.Embedding(100, atom_fea_len)
        self.emb_proj = nn.Linear(atom_fea_len, self.hidden)
        self.convs = nn.ModuleList(
            [CGCNNLayer(self.hidden, self.rbf.out_dim, use_bn=use_bn) for _ in range(n_conv)]
        )
        self.fc_out = nn.Linear(self.hidden, n_classes)

        if use_jk:
            self.jk_proj = nn.Linear((n_conv + 1) * self.hidden, self.hidden)

        head_in_mult = 1
        if use_shell_attn:
            self.shell_q = nn.Linear(self.hidden, self.hidden)
            self.shell_k = nn.Linear(self.hidden, self.hidden)
            self.shell_d = nn.Linear(self.rbf.out_dim, self.hidden)
            self.shell_v = nn.Linear(self.hidden, self.hidden)
            head_in_mult += 1

        if use_so3_geo:
            self.geo_w = nn.Linear(self.rbf.out_dim, 1)
            self.geo_proj = nn.Linear(4, self.hidden)
            head_in_mult += 1

        self.head_lin = nn.Linear(head_in_mult * self.hidden, self.hidden)
        if use_res_temp:
            self.head_norm = nn.LayerNorm(self.hidden)
            self.head_lin2 = nn.Linear(self.hidden, self.hidden)
            self.log_T = nn.Parameter(torch.zeros(1))

    def _shell_geometry(self, nbr_dists, target_index, nbr_vecs, batch_idx):
        B = nbr_dists.shape[0]
        T = target_index.shape[-1]
        dev = nbr_dists.device
        if nbr_vecs is None:
            return torch.zeros(B, T, 4, device=dev)
        d_t = nbr_dists[batch_idx[:, None], target_index]
        u_t = nbr_vecs[batch_idx[:, None], target_index]
        u_t = u_t / (u_t.norm(dim=-1, keepdim=True) + 1e-6)
        w = torch.softmax(self.geo_w(self.rbf(d_t)), dim=2).squeeze(-1)
        dvec = (w.unsqueeze(-1) * u_t).sum(2)
        dip = dvec.norm(dim=-1)
        outer = u_t.unsqueeze(-1) * u_t.unsqueeze(-2)
        Q = (w.unsqueeze(-1).unsqueeze(-1) * outer).sum(2)
        tr = torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1)
        Qdev = Q - (tr / 3.0).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=dev)
        quad = (Qdev ** 2).sum(dim=(-1, -2)).sqrt()
        C = u_t @ u_t.transpose(-1, -2)
        S2 = torch.einsum("btj,btjk,btk->bt", w, C ** 2, w)
        S3 = torch.einsum("btj,btjk,btk->bt", w, C ** 3, w)
        return torch.stack([dip, quad, S2, S3], dim=-1)

    def forward(
        self,
        atom_types: torch.Tensor,
        nbr_dists: torch.Tensor,
        nbr_lists: torch.Tensor,
        target_index: torch.Tensor,
        nbr_vecs: torch.Tensor = None,
    ) -> torch.Tensor:
        atom_fea = F.silu(self.emb_proj(self.embedding(atom_types)))
        bond_fea = self.rbf(nbr_dists)
        feats = [atom_fea]
        for conv in self.convs:
            atom_fea = conv(atom_fea, bond_fea, nbr_lists)
            feats.append(atom_fea)
        jk = F.silu(self.jk_proj(torch.cat(feats, dim=-1))) if self.use_jk else atom_fea
        batch_idx = torch.arange(jk.shape[0], device=jk.device)
        target = jk[batch_idx[:, None], target_index]

        parts = [target]
        if self.use_shell_attn:
            nl_t = nbr_lists[batch_idx[:, None], target_index]
            nbr_t = jk[batch_idx[:, None, None], nl_t]
            d_t = nbr_dists[batch_idx[:, None], target_index]
            q = self.shell_q(target).unsqueeze(2)
            k = self.shell_k(nbr_t) + self.shell_d(self.rbf(d_t))
            attn = torch.softmax((q * k).sum(-1) / (self.hidden ** 0.5), dim=2)
            shell = (attn.unsqueeze(-1) * self.shell_v(nbr_t)).sum(2)
            parts.append(shell)
        if self.use_so3_geo:
            geo = F.silu(self.geo_proj(self._shell_geometry(nbr_dists, target_index, nbr_vecs, batch_idx)))
            parts.append(geo)

        feat = F.silu(self.head_lin(torch.cat(parts, dim=-1)))
        if self.use_res_temp:
            feat = feat + F.silu(self.head_lin2(self.head_norm(feat)))
            temperature = torch.clamp(torch.exp(self.log_T), 0.3, 1.5)
            return F.softmax(self.fc_out(feat) / temperature, dim=-1)
        return F.softmax(self.fc_out(feat), dim=-1)


VARIANTS = {
    "canonical_h16": {"kind": "canonical", "atom_fea_len": 16},
    "canonical_pm_h46": {"kind": "canonical", "atom_fea_len": 46},
    "geoshell_h16": {"kind": "geoshell", "atom_fea_len": 8, "width_mult": 2},
    "geoshell_h32": {"kind": "geoshell", "atom_fea_len": 16, "width_mult": 2},
    "wo_so3_geo": {"kind": "geoshell", "atom_fea_len": 16, "width_mult": 2, "use_so3_geo": False},
    "wo_shell_attn": {"kind": "geoshell", "atom_fea_len": 16, "width_mult": 2, "use_shell_attn": False},
    "wo_jk": {"kind": "geoshell", "atom_fea_len": 16, "width_mult": 2, "use_jk": False},
    "wo_res_temp": {"kind": "geoshell", "atom_fea_len": 16, "width_mult": 2, "use_res_temp": False},
    "head_only_no_jk_no_attn": {"kind": "geoshell", "atom_fea_len": 16, "width_mult": 2, "use_jk": False, "use_shell_attn": False},
    "jk_only_no_attn_no_geo": {"kind": "geoshell", "atom_fea_len": 16, "width_mult": 2, "use_shell_attn": False, "use_so3_geo": False},
    "width_h64": {"kind": "geoshell", "atom_fea_len": 16, "width_mult": 4},
}


def build_variant_model(variant: str = "geoshell_h32", n_classes: int = 4, n_conv: int = 3, use_bn: bool = True) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}. Valid: {list(VARIANTS)}")
    cfg = dict(VARIANTS[variant])
    kind = cfg.pop("kind")
    if kind == "canonical":
        return CanonicalGDyNet(n_classes=n_classes, n_conv=n_conv, use_bn=use_bn, **cfg)
    return GeoShellGDyNet(n_classes=n_classes, n_conv=n_conv, use_bn=use_bn, **cfg)
