# strategy_engine_combined.py

"""
Combined architecture + decision engine + execution engine.

- FullModelOptimized: multi-force encoders -> vectorized interaction -> temporal encoder -> EV head -> tail module
- DecisionEngine: EV/R/Kelly-based entry/management logic (from user's provided code; cleaned)
- ExecutionEngine: translates decisions into execution actions and updates tail buffer with realized losses

No synthetic demos.
PCA-free: preprocessors must supply final (per_force_in) features.
"""

import math
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------


def make_causal_force_time_mask(T: int, N: int, device=None) -> torch.BoolTensor:
    L = T * N
    idx = torch.arange(L, device=device)
    t_idx = idx // N
    t_i = t_idx.unsqueeze(1)
    t_j = t_idx.unsqueeze(0)
    mask = (t_j > t_i)
    return mask


def robust_std(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 0.0
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return float(1.4826 * mad + 1e-12)


def normalize_reliability_weights(
    r: Optional[torch.Tensor], B: int, T: int, N: int, device
) -> Optional[torch.Tensor]:
    if r is None:
        return None
    if not isinstance(r, torch.Tensor):
        r = torch.tensor(r, device=device, dtype=torch.float32)

    if r.dim() == 1:
        if r.shape[0] != N:
            raise ValueError(f"1D reliability must be length N={N}")
        return r.view(1, 1, N).expand(B, T, N).to(device)

    if r.dim() == 2:
        if r.shape[1] != N:
            raise ValueError(f"2D reliability must have shape (B, N) with N={N}")
        if r.shape[0] not in (B,):
            if r.shape[0] == 1:
                return r.view(1, 1, N).expand(B, T, N).to(device)
            raise ValueError(
                f"2D reliability shape mismatch: expected (B,N) got {tuple(r.shape)}"
            )
        return r.view(B, 1, N).expand(B, T, N).to(device)

    if r.dim() == 3:
        if tuple(r.shape) != (B, T, N):
            raise ValueError(
                f"3D reliability must be (B,T,N). got {tuple(r.shape)}"
            )
        return r.to(device)

    raise ValueError(f"Unsupported reliability_weights shape {tuple(r.shape)}")


# -------------------------
# PerForceGating
# -------------------------


class PerForceGating(nn.Module):
    def __init__(self, force_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(force_dim),
            nn.Linear(force_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, per_force_emb: torch.Tensor) -> torch.Tensor:
        logits = self.net(per_force_emb).squeeze(-1)
        return torch.sigmoid(logits)


# -------------------------
# ForceModule
# -------------------------


class ForceModule(nn.Module):
    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, T, C = x.shape
            # use reshape for safety with potential non-contiguous inputs
            xf = x.reshape(B * T, C)
            xf = self.norm(xf)
            h = self.net(xf)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=self.training)
            return self.proj(h).view(B, T, -1)
        else:
            x2 = self.norm(x)
            h = self.net(x2)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=self.training)
            return self.proj(h)


# -------------------------
# VectorizedInteraction
# -------------------------


class VectorizedInteraction(nn.Module):
    def __init__(
        self,
        n_forces: int = 6,
        force_dim: int = 32,
        hidden: int = 128,
        final_dim: int = 128,
        n_heads: int = 4,
        use_gating: bool = True,
        dropout: float = 0.1,
        softstart_steps: int = 3000,
    ):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_forces = n_forces
        self.force_dim = force_dim
        self.hidden = hidden
        self.final_dim = final_dim
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.use_gating = use_gating
        self.dropout = dropout
        self.softstart_steps = softstart_steps

        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

        self.q_proj = nn.Linear(force_dim, hidden)
        self.k_proj = nn.Linear(force_dim, hidden)
        self.v_proj = nn.Linear(force_dim, hidden)

        if use_gating:
            self.gater = PerForceGating(force_dim, hidden=max(32, force_dim))

        self.out_proj = nn.Linear(hidden, hidden)
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, final_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        per_t_forces: torch.Tensor,
        T: int,
        reliability_weights: Optional[torch.Tensor] = None,
        debug: bool = False,
    ):
        B, TT, N, D = per_t_forces.shape
        assert TT == T and N == self.n_forces and D == self.force_dim

        if self.training:
            with torch.no_grad():
                self.training_step.add_(1)

        device = per_t_forces.device
        L = T * N
        x = per_t_forces.view(B, L, D)

        if self.use_gating:
            learned_gates = self.gater(x).view(B, T, N)
            if self.training and int(self.training_step.item()) < self.softstart_steps:
                learned_gates = learned_gates.detach()
        else:
            learned_gates = torch.ones(B, T, N, device=device, dtype=x.dtype)

        r = (
            normalize_reliability_weights(
                reliability_weights, B, T, N, device
            )
            if reliability_weights is not None
            else None
        )

        if r is not None:
            final_gates = torch.sqrt(learned_gates * r)
        else:
            final_gates = learned_gates

        final_gates_flat = final_gates.view(B, L).unsqueeze(-1)

        Q = self.q_proj(x) * final_gates_flat
        K = self.k_proj(x) * final_gates_flat
        V = self.v_proj(x) * final_gates_flat

        def split_heads(z):
            return (
                z.view(B, L, self.n_heads, self.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = make_causal_force_time_mask(T, N, device=device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        A = torch.softmax(scores, dim=-1).clamp(min=1e-9)
        attn_out = torch.matmul(A, Vh)

        attn_concat = (
            attn_out.permute(0, 2, 1, 3).reshape(B, L, self.hidden)
        )
        attn_concat = F.dropout(
            attn_concat, p=self.dropout, training=self.training
        )

        out = self.out_proj(attn_concat)
        out_r = out.view(B, T, N, self.hidden)
        per_t_out = out_r.mean(dim=2)

        final = self.final_proj(per_t_out)

        diag = {
            "learned_gates": learned_gates.detach().cpu(),
            "final_gates": final_gates.detach().cpu(),
        }
        if debug:
            diag["attention_head_mean"] = A.mean(dim=-1).detach().cpu()

        return final, diag


# -------------------------
# TemporalTransformerEncoder
# -------------------------


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        ff_hidden: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.register_buffer("pos_emb", None)

    def _build_pos_emb(self, T: int, D: int, device):
        pos = torch.arange(0, T, device=device).unsqueeze(1)
        i = torch.arange(0, D // 2, device=device).unsqueeze(0)
        angle_rates = 1.0 / (10000 ** (2.0 * i.float() / float(D)))
        angle_rads = pos.float() * angle_rates
        pe = torch.zeros(1, T, D, device=device)
        pe[0, :, 0::2] = torch.sin(angle_rads)
        pe[0, :, 1::2] = torch.cos(
            angle_rads[:, : pe[0, :, 1::2].shape[1]]
        )
        return pe

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        device = x.device

        if self.pos_emb is None or self.pos_emb.shape[1] < T:
            pe = self._build_pos_emb(max(T, 8), D, device)
            self.pos_emb = pe

        x = x + self.pos_emb[:, :T, :].to(device)
        attn_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        out = self.encoder(x, mask=attn_mask)
        return out


# -------------------------
# TemporalEVHead
# -------------------------


class TemporalEVHead(nn.Module):
    def __init__(
        self,
        inter_dim: int = 128,
        horizons: Optional[List[int]] = None,
        attn_hidden: int = 64,
        out_hidden: int = 64,
    ):
        super().__init__()
        if horizons is None:
            horizons = [1, 5, 20]
        self.horizons = horizons
        self.H = len(horizons)

        self.attn_proj = nn.Sequential(
            nn.Linear(inter_dim, attn_hidden),
            nn.Tanh(),
        )
        self.pool_w = nn.Parameter(torch.randn(attn_hidden) * 0.01)

        self.mean_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(inter_dim),
                    nn.Linear(inter_dim, out_hidden),
                    nn.SiLU(),
                    nn.Linear(out_hidden, 1),
                )
                for _ in range(self.H)
            ]
        )

        self.logvar_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(inter_dim),
                    nn.Linear(inter_dim, out_hidden),
                    nn.SiLU(),
                    nn.Linear(out_hidden, 1),
                )
                for _ in range(self.H)
            ]
        )

        self.conf_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(inter_dim),
                    nn.Linear(inter_dim, out_hidden),
                    nn.SiLU(),
                    nn.Linear(out_hidden, 1),
                )
                for _ in range(self.H)
            ]
        )

    def attention_pool(self, seq: torch.Tensor) -> torch.Tensor:
        proj = self.attn_proj(seq)
        scores = torch.matmul(proj, self.pool_w)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (alpha * seq).sum(dim=1)
        return pooled

    def forward(
        self, seq_h_inter: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.attention_pool(seq_h_inter)

        mus, sigmas, confs = [], [], []
        for i in range(self.H):
            mu = self.mean_heads[i](pooled).squeeze(-1)
            logvar = self.logvar_heads[i](pooled).squeeze(-1)
            sigma = torch.sqrt(F.softplus(logvar) + 1e-8)
            conf = torch.sigmoid(
                self.conf_heads[i](pooled).squeeze(-1)
            )
            mus.append(mu)
            sigmas.append(sigma)
            confs.append(conf)

        mu = torch.stack(mus, dim=1)
        sigma = torch.stack(sigmas, dim=1)
        conf = torch.stack(confs, dim=1)
        return mu, sigma, conf

    @staticmethod
    def nll_loss(
        mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        var = sigma.pow(2) + 1e-8
        nll = (
            0.5 * ((target - mu).pow(2) / var)
            + 0.5 * torch.log(var)
            + 0.5 * math.log(2 * math.pi)
        )
        return nll.mean()


# -------------------------
# TailRiskModule
# -------------------------


class TailRiskModule(nn.Module):
    """
    Decoupled tail risk estimator.

    Design goals:
    - Do NOT depend directly on model EV/sigma (no self-referential risk).
    - Use a simple NN on the final hidden state to estimate a baseline tail probability.
    - Combine that with an empirical loss buffer in a *smooth* way (no max(), no cliffs).
    - Keep API compatible: forward(h_inter_last, ev_sigma) still accepted, but ev_sigma ignored.
    """

    def __init__(
        self,
        inter_dim: int = 128,
        buffer_size: int = 2000,
        pot_quantile: float = 0.95,
        min_samples: int = 50,
        smooth_blend: float = 0.3,  # weight on empirical tail vs NN tail
    ):
        super().__init__()
        self.register_buffer(
            "loss_buffer", torch.zeros(buffer_size, dtype=torch.float32)
        )
        self.register_buffer("buffer_ptr", torch.tensor(0, dtype=torch.long))
        self.buffer_size = buffer_size
        self.pot_quantile = pot_quantile
        self.min_samples = max(10, min_samples)
        self.inter_dim = inter_dim
        self.smooth_blend = float(np.clip(smooth_blend, 0.0, 1.0))

        # NN only sees representation, not EV/sigma
        self.nn_head = nn.Sequential(
            nn.LayerNorm(inter_dim),
            nn.Linear(inter_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    @torch.jit.ignore
    def update_losses(self, losses: torch.Tensor):
        """
        Accepts positive loss magnitudes (already abs()'d).
        Maintains a rolling buffer without gradients.
        """
        losses = losses.detach().cpu().float()
        n = losses.numel()
        if n == 0:
            return

        ptr = int(self.buffer_ptr.item())
        size = self.buffer_size

        if n >= size:
            self.loss_buffer.copy_(losses[-size:])
            self.buffer_ptr.fill_(0)
        else:
            end = ptr + n
            if end <= size:
                self.loss_buffer[ptr:end] = losses
                self.buffer_ptr.fill_(end % size)
            else:
                first = size - ptr
                self.loss_buffer[ptr:] = losses[:first]
                self.loss_buffer[: (n - first)] = losses[first:]
                self.buffer_ptr.fill_(n - first)

    def empirical_tail_stats(self):
        """
        Simple empirical exceedance stats.
        No GPD fitting: we're using this only as a slow, smoothed sanity check.
        """
        arr = self.loss_buffer.detach().cpu().numpy()
        arr_nonzero = arr[arr > 0.0]
        if arr_nonzero.size < self.min_samples:
            return 0.0, 0.0, 0.0, 0

        u = float(np.quantile(arr_nonzero, self.pot_quantile))
        exceed = arr_nonzero[arr_nonzero >= u]
        k = exceed.size
        n = arr_nonzero.size
        emp_p = float(k / n)
        emp_cvar = float(exceed.mean()) if k > 0 else 0.0
        return emp_p, emp_cvar, u, k

    def forward(
        self, h_inter_last: torch.Tensor, ev_sigma: torch.Tensor
    ) -> Dict[str, Any]:
        """
        h_inter_last: (B, inter_dim)
        ev_sigma: kept for API compatibility, but ignored for risk estimation.
        """
        B = h_inter_last.shape[0]
        device = h_inter_last.device

        # NN-based tail probability in [0,1]
        tail_nn = torch.sigmoid(self.nn_head(h_inter_last).squeeze(-1))

        # Empirical tail probability from loss history
        emp_p, emp_cvar, u, k = self.empirical_tail_stats()
        emp_p_t = torch.tensor(emp_p, device=device, dtype=tail_nn.dtype)

        # Smooth blend: avoid discontinuities, keep empirical as slow anchor
        combined = (1.0 - self.smooth_blend) * tail_nn + self.smooth_blend * emp_p_t

        # Optional inflation based on average tail size (smaller than before)
        if emp_cvar > 0:
            inflation = 0.1 * math.tanh(emp_cvar)
            combined = torch.clamp(combined + inflation, 0.0, 0.999)

        return {
            "tail_nn": tail_nn.detach().cpu(),
            "tail_emp_p": emp_p,
            "tail_emp_cvar": emp_cvar,
            "tail_threshold_u": u,
            "tail_n_exceed": k,
            "tail_combined": combined.detach().cpu(),
        }



# -------------------------
# FullModelOptimized
# -------------------------


class FullModelOptimized(nn.Module):
    """
    Patched version with explicit role separation for forces:

    - initiation forces: typically OFI, LiquidityHunt, VolShock
    - propagation forces: trend-like
    - stabilization forces: mean-revert, inventory pressure

    We:
    - keep the external API the same: forward(x_seq, reliability) -> dict
    - internally run three VectorizedInteraction blocks (one per role)
    - concatenate their outputs before the temporal encoder and EV head
    """

    def __init__(
        self,
        n_forces: int = 6,
        per_force_in: int = 32,
        per_force_hidden: int = 64,
        per_force_out: int = 32,
        interaction_dim_per_role: int = 64,
        n_heads: int = 4,
        transformer_layers: int = 2,
        horizons: Optional[List[int]] = None,
        role_assignment: Optional[List[str]] = None,
    ):
        super().__init__()
        self.n_forces = n_forces
        self.per_force_out = per_force_out

        # Role assignment: length n_forces, each in {"init","prop","stab"}
        if role_assignment is None:
            if n_forces != 6:
                raise ValueError(
                    "Default role_assignment assumes n_forces=6 "
                    "(OFI, LiquidityHunt, VolShock, Trend, MeanRevert, Inventory). "
                    "Please pass an explicit role_assignment list."
                )
            # Default mapping for your described setup:
            # 0: OFI, 1: LiquidityHunt, 2: VolShock → initiation
            # 3: Trend → propagation
            # 4: MeanRevert, 5: InventoryPressure → stabilization
            role_assignment = ["init", "init", "init", "prop", "stab", "stab"]

        if len(role_assignment) != n_forces:
            raise ValueError("role_assignment length must equal n_forces")

        self.role_assignment = role_assignment

        # Per-force encoders (unchanged idea)
        self.force_modules = nn.ModuleList(
            [
                ForceModule(
                    per_force_in,
                    per_force_hidden,
                    per_force_out,
                )
                for _ in range(n_forces)
            ]
        )

        # indices per role for slicing
        self.init_idx = [i for i, r in enumerate(role_assignment) if r == "init"]
        self.prop_idx = [i for i, r in enumerate(role_assignment) if r == "prop"]
        self.stab_idx = [i for i, r in enumerate(role_assignment) if r == "stab"]

        if not self.init_idx or not self.prop_idx or not self.stab_idx:
            raise ValueError(
                f"Each role must have at least one force. Got init={self.init_idx}, "
                f"prop={self.prop_idx}, stab={self.stab_idx}"
            )

        # one interaction block per role
        self.interaction_init = VectorizedInteraction(
            n_forces=len(self.init_idx),
            force_dim=per_force_out,
            hidden=interaction_dim_per_role,
            final_dim=interaction_dim_per_role,
            n_heads=n_heads,
            use_gating=True,
            dropout=0.1,
            softstart_steps=3000,
        )
        self.interaction_prop = VectorizedInteraction(
            n_forces=len(self.prop_idx),
            force_dim=per_force_out,
            hidden=interaction_dim_per_role,
            final_dim=interaction_dim_per_role,
            n_heads=n_heads,
            use_gating=True,
            dropout=0.1,
            softstart_steps=3000,
        )
        self.interaction_stab = VectorizedInteraction(
            n_forces=len(self.stab_idx),
            force_dim=per_force_out,
            hidden=interaction_dim_per_role,
            final_dim=interaction_dim_per_role,
            n_heads=n_heads,
            use_gating=True,
            dropout=0.1,
            softstart_steps=3000,
        )

        # total interaction dim is sum of per-role dims
        interaction_dim_total = 3 * interaction_dim_per_role

        self.temporal = TemporalTransformerEncoder(
            d_model=interaction_dim_total,
            n_heads=n_heads,
            n_layers=transformer_layers,
        )

        self.ev_head = TemporalEVHead(
            inter_dim=interaction_dim_total, horizons=horizons
        )

        self.tail_module = TailRiskModule(
            inter_dim=interaction_dim_total,
            buffer_size=2000,
            pot_quantile=0.95,
            min_samples=50,
        )

        self.register_buffer("_ema_count", torch.tensor(0, dtype=torch.long))
        self.ema_shadow: Dict[str, torch.Tensor] = {}

    def forward(
        self, x_seq: torch.Tensor, reliability: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        x_seq: (B,T,N,F_in) with N=n_forces
        reliability: optional reliability weights per force (same semantics as before)
        """
        B, T, N, F_in = x_seq.shape
        assert N == self.n_forces

        # 1) per-force encoders (unchanged logic)
        per_list = []
        for i in range(self.n_forces):
            per = self.force_modules[i](x_seq[:, :, i, :])
            per_list.append(per)  # (B,T,per_force_out)

        per_t = torch.stack(per_list, dim=2)  # (B,T,N,per_force_out)

        # 2) split into roles
        per_init = per_t[:, :, self.init_idx, :]  # (B,T,N_init,D)
        per_prop = per_t[:, :, self.prop_idx, :]  # (B,T,N_prop,D)
        per_stab = per_t[:, :, self.stab_idx, :]  # (B,T,N_stab,D)

        # optional: split reliability by role if provided
        rel_init = rel_prop = rel_stab = None
        if reliability is not None:
            # reliability expected shape is (B,T,N)
            rel_init = reliability[:, :, self.init_idx]
            rel_prop = reliability[:, :, self.prop_idx]
            rel_stab = reliability[:, :, self.stab_idx]

        # 3) run interactions per role
        inter_init, diag_init = self.interaction_init(
            per_init, T, reliability_weights=rel_init, debug=False
        )  # (B,T,D_role)
        inter_prop, diag_prop = self.interaction_prop(
            per_prop, T, reliability_weights=rel_prop, debug=False
        )
        inter_stab, diag_stab = self.interaction_stab(
            per_stab, T, reliability_weights=rel_stab, debug=False
        )

        # 4) concatenate along feature dimension → (B,T,3*D_role)
        inter_seq = torch.cat([inter_init, inter_prop, inter_stab], dim=-1)

        # 5) temporal encoder + EV head
        inter_seq_enc = self.temporal(inter_seq)
        mu_multi, sigma_multi, conf_multi = self.ev_head(inter_seq_enc)

        ev_mean = mu_multi[:, 0]
        ev_sigma = sigma_multi[:, 0]
        conf = conf_multi[:, 0]

        # 6) tail module (decoupled from EV sigma internally, but API preserved)
        tail_out = self.tail_module(
            inter_seq_enc[:, -1, :], ev_sigma
        )

        out = {
            "per_force_emb": per_t.detach().cpu(),
            "h_inter_seq": inter_seq_enc,
            "ev_mean_multi": mu_multi,
            "ev_sigma_multi": sigma_multi,
            "conf_multi": conf_multi,
            "ev_mean": ev_mean,
            "ev_sigma": ev_sigma,
            "conf": conf,
            "tail": tail_out,
            "interaction_diag": {
                "init": diag_init,
                "prop": diag_prop,
                "stab": diag_stab,
            },
        }
        return out



# -------------------------
# DecisionEngine
# -------------------------


class DecisionEngine:
    """
    EV -> Kelly final sizing decision engine.

    Accepts arrays (older->newer) for:
    ev_hist, conf_hist, tail_hist, kelly_soft_hist, qtp_hist,
    hazard_hist, forces_hist.
    """

    def __init__(
        self,
        history_len: int = 12,
        ev_cluster_len: int = 5,
        ewma_alpha: float = 0.4,
        ev_min_abs: float = 0.0020,
        ev_min_sigma_mul: float = 0.6,
        conf_entry: float = 0.70,
        conf_hold: float = 0.45,
        qtp_entry: float = 0.50,
        qtp_hold: float = 0.30,
        tail_max: float = 0.12,
        hazard_cum_thresh: float = 0.35,
        force_align_min: int = 4,
        lambda_size: float = 3.0,
        fmax: float = 0.02,
        lock_size_on_entry: bool = True,
        no_pyramiding: bool = True,
        allow_controlled_pyramid: bool = False,
        pyramid_max_legs: int = 2,
        pyramid_ev_increase_pct: float = 0.35,
        max_hold_bars: int = 48,
        partial_trim_fraction: float = 0.5,
        costs_margin_buffer: float = 1e-9,
        debug_invariants: bool = True,
    ):
        self.H = history_len
        self.ev_cluster_len = ev_cluster_len
        self.ewma_alpha = ewma_alpha
        self.ev_min_abs = ev_min_abs
        self.ev_min_sigma_mul = ev_min_sigma_mul
        self.conf_entry = conf_entry
        self.conf_hold = conf_hold
        self.qtp_entry = qtp_entry
        self.qtp_hold = qtp_hold
        self.tail_max = tail_max
        self.hazard_cum_thresh = hazard_cum_thresh
        self.force_align_min = force_align_min
        self.lambda_size = lambda_size
        self.fmax = fmax
        self.lock_size_on_entry = lock_size_on_entry
        self.no_pyramiding = no_pyramiding
        self.allow_controlled_pyramid = allow_controlled_pyramid
        self.pyramid_max_legs = pyramid_max_legs
        self.pyramid_ev_increase_pct = pyramid_ev_increase_pct
        self.max_hold_bars = max_hold_bars
        self.partial_trim_fraction = partial_trim_fraction
        self.costs_margin_buffer = costs_margin_buffer

        self.debug_invariants = debug_invariants
        self.current_position = None

    # ---------------- helper utils ----------------

    def ewma(self, arr, alpha):
        s = None
        for v in arr:
            s = v if s is None else alpha * v + (1 - alpha) * s
        return s if s is not None else 0.0

    def rolling_mean(self, arr, n):
        if len(arr) == 0:
            return 0.0
        arrn = np.array(arr)
        return arrn[-n:].mean() if len(arrn) >= n else arrn.mean()

    def ev_cluster(self, ev_hist):
        recent = ev_hist[-self.ev_cluster_len :]
        ew = self.ewma(recent, self.ewma_alpha)
        rm = self.rolling_mean(ev_hist, self.ev_cluster_len)
        return 0.6 * ew + 0.4 * rm

    def sigma_noise(self, ev_hist):
        arr = np.array(ev_hist[-max(2, self.ev_cluster_len) :])
        if arr.size < 2:
            return 1e-9
        diffs = np.diff(arr)
        return max(float(np.std(diffs, ddof=0)), 1e-9)

    def compute_force_alignment(self, forces_latest):
        if forces_latest is None:
            return 0, 1, 0.0
        arr = np.array(forces_latest)
        f = arr[-1] if arr.ndim == 2 else arr
        signs = np.sign(f)
        mags = np.abs(f)
        median_mag = np.median(mags) + 1e-9
        active = mags >= (0.5 * median_mag)
        pos = int(np.sum((signs > 0) & active))
        neg = int(np.sum((signs < 0) & active))
        dominant_sign = 1 if pos >= neg else -1
        align_count = max(pos, neg)
        avg_mag = float(np.mean(mags[active])) if np.any(active) else 0.0
        return align_count, dominant_sign, avg_mag

    def cumulative_hazard(self, hazard_hist, m=6):
        if hazard_hist is None or len(hazard_hist) == 0:
            return 0.0
        short = np.array(hazard_hist[-m:])
        return float(1.0 - np.prod(1.0 - short))

    def ev_price_to_R(self, ev_price, sl_price_dist):
        return 0.0 if sl_price_dist <= 0 else ev_price / sl_price_dist

    def cost_fraction_in_R(self, spread_price, sl_price_dist, estimated_slippage_price=0.0):
        if sl_price_dist <= 0:
            return 1e9
        return (
            spread_price
            + estimated_slippage_price
            + self.costs_margin_buffer
        ) / sl_price_dist

    # ---------------- invariants ----------------

    def _validate_position_state(self):
        if not self.debug_invariants or self.current_position is None:
            return
        pos = self.current_position
        assert pos["side"] in (-1, 1)
        assert pos["size"] >= 0.0
        if pos.get("locked_size") is not None:
            assert pos["locked_size"] >= 0.0
        assert isinstance(pos["entry_bar"], int)
        assert isinstance(pos.get("legs", 1), int) and pos["legs"] >= 1

    # ---------------- main decision API ----------------

    def decide(
        self,
        *,
        ev_hist,
        conf_hist,
        tail_hist,
        kelly_soft_hist,
        qtp_hist,
        hazard_hist,
        forces_hist,
        spread_price,
        estimated_slippage_price,
        sl_price_dist,
        current_bar_index,
    ):

        if len(ev_hist) == 0:
            return {"action": "no_action", "reason": "no_data", "details": {}}

        ev_pred_latest = float(ev_hist[-1])
        conf_latest = float(conf_hist[-1])
        tail_latest = float(tail_hist[-1])
        kelly_soft_latest = float(kelly_soft_hist[-1]) if kelly_soft_hist else 0.0
        qtp_latest = float(qtp_hist[-1]) if qtp_hist else 0.0

        # ===== REPLACED SECTION STARTS HERE =====

        ev_clust = self.ev_cluster(ev_hist)
        sigma = self.sigma_noise(ev_hist)
        ev_min_adaptive = max(self.ev_min_abs, self.ev_min_sigma_mul * sigma)

        cum_hazard = self.cumulative_hazard(hazard_hist, m=6)
        align_count, dominant_sign, avg_force_mag = (
            self.compute_force_alignment(forces_hist)
        )

        ev_R = self.ev_price_to_R(ev_pred_latest, sl_price_dist)
        cost_R = self.cost_fraction_in_R(
            spread_price, sl_price_dist, estimated_slippage_price
        )
        ev_R_adj = ev_R - cost_R
        ev_final = ev_R_adj * conf_latest

        diagnostics = dict(
            ev_pred=float(ev_pred_latest),
            ev_cluster=float(ev_clust),
            ev_R=float(ev_R),
            cost_R=float(cost_R),
            ev_R_adj=float(ev_R_adj),
            ev_final=float(ev_final),
            conf=float(conf_latest),
            qtp=float(qtp_latest),
            tail=float(tail_latest),
            cum_hazard=float(cum_hazard),
            align_count=int(align_count),
            dominant_sign=int(dominant_sign),
            avg_force_mag=float(avg_force_mag),
            locked_position=bool(self.current_position is not None),
        )

        # ENTRY (simplified, less coupled)
        if self.current_position is None:
            gate_ev_cluster = ev_clust >= ev_min_adaptive and ev_R_adj > 0
            gate_conf = conf_latest >= self.conf_entry
            gate_qtp = qtp_latest >= self.qtp_entry
            gate_hazard = cum_hazard <= self.hazard_cum_thresh
            gate_align = align_count >= self.force_align_min

            if gate_ev_cluster and gate_conf and gate_qtp and gate_hazard and gate_align:
                side = 1 if ev_clust > 0 else -1
                if side != dominant_sign:
                    side = 1 if ev_clust > 0 else -1

                f_raw = self.lambda_size * ev_final
                f_final = float(max(0.0, min(self.fmax, f_raw)))

                if f_final <= 0 and kelly_soft_latest > 0:
                    f_final = float(min(self.fmax, kelly_soft_latest * 0.25))

                self.current_position = {
                    "side": int(side),
                    "size": float(f_final),
                    "locked_size": float(f_final)
                    if self.lock_size_on_entry
                    else None,
                    "entry_bar": int(current_bar_index),
                    "legs": 1,
                    "entry_ev_R": float(ev_R),
                    "entry_ev_final": float(ev_final),
                    "entry_conf": float(conf_latest),
                    "entry_qtp": float(qtp_latest),
                }

                self._validate_position_state()

                return {
                    "action": "enter_long" if side > 0 else "enter_short",
                    "side": int(side),
                    "size": float(f_final),
                    "reason": "entry_gates_passed",
                    "details": diagnostics,
                }

            return {
                "action": "no_action",
                "reason": "entry_gates_failed",
                "details": diagnostics,
            }

        # ===== EVERYTHING BELOW UNCHANGED =====

        pos = self.current_position
        self._validate_position_state()
        side = pos["side"]
        locked_size = pos["locked_size"] if pos.get("locked_size") is not None else pos["size"]
        bars_held = current_bar_index - pos["entry_bar"]

        if tail_latest > (self.tail_max * 1.25):
            self._clear_position()
            return {"action": "exit", "side": -side, "size": locked_size, "reason": "exit_tail_spike", "details": diagnostics}

        if cum_hazard > (self.hazard_cum_thresh * 1.25):
            self._clear_position()
            return {"action": "exit", "side": -side, "size": locked_size, "reason": "exit_hazard_spike", "details": diagnostics}

        if conf_latest < self.conf_hold and abs(ev_clust) < 0.5 * abs(pos["entry_ev_R"]):
            self._clear_position()
            return {"action": "exit", "side": -side, "size": locked_size, "reason": "exit_confidence_collapsed", "details": diagnostics}

        if side * ev_R <= 0:
            self._clear_position()
            return {"action": "exit", "side": -side, "size": locked_size, "reason": "exit_ev_flip", "details": diagnostics}

        if side * ev_R < 0.25 * pos["entry_ev_R"]:
            trim = locked_size * self.partial_trim_fraction
            remaining = locked_size - trim
            if remaining <= 0:
                self._clear_position()
            else:
                pos["size"] = remaining
                if pos.get("locked_size") is not None:
                    pos["locked_size"] = remaining
                self._validate_position_state()
            return {"action": "partial_exit", "side": -side, "size": trim, "reason": "partial_trim_ev_decay", "details": diagnostics}

        if bars_held >= self.max_hold_bars:
            self._clear_position()
            return {"action": "exit", "side": -side, "size": locked_size, "reason": "exit_max_hold", "details": diagnostics}

        return {"action": "hold", "side": side, "size": locked_size, "reason": "hold", "details": diagnostics}

    def _clear_position(self):
        self.current_position = None


#RISK SUPERVISOR
class RiskSupervisor:
    def __init__(self, base_fmax, pnl_regime_threshold=-0.01, loss_pause_bars=10):
        self.base_fmax = base_fmax
        self.current_fmax = base_fmax
        self.pnl_regime_threshold = pnl_regime_threshold
        self.loss_pause_bars = loss_pause_bars
        self.consecutive_losses = 0
        self.pause_until_bar = -1

    def update_regime(self, rolling_pnl_30m: float):
        if rolling_pnl_30m < self.pnl_regime_threshold:
            self.current_fmax = self.base_fmax * 0.25
        else:
            self.current_fmax = self.base_fmax

    # MUST BE INDENTED LIKE THIS
    def apply_tail_throttle(self, tail_combined: float):
        if tail_combined > 0.12:
            self.current_fmax *= 0.25
        elif tail_combined > 0.08:
            self.current_fmax *= 0.50

    def on_trade_closed(self, realized_pnl: float, current_bar: int):
        if realized_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= 2:
            self.pause_until_bar = current_bar + self.loss_pause_bars

    def entries_allowed(self, current_bar: int) -> bool:
        return current_bar >= self.pause_until_bar


# -------------------------
# ExecutionEngine
# -------------------------


class ExecutionEngine:
    """
    Minimal execution engine that maps decision -> order dict and updates tail buffer.

    Usage:
        exec_engine = ExecutionEngine(model, decision_engine, risk_supervisor=...)
        exec_engine.step(model_output, market_prices, current_bar)

    This class does NOT connect to any exchange — it returns order dicts that your adapter should send.
    """

    def __init__(
        self,
        model: FullModelOptimized,
        decision: DecisionEngine,
        min_size_frac: float = 1e-6,
        risk_supervisor: Optional["RiskSupervisor"] = None,  # risk layer
    ):
        self.model = model
        self.decision = decision
        self.min_size_frac = min_size_frac
        self.risk = risk_supervisor

    def _make_order(
        self, side: int, size_frac: float, market_price: float, note: str = ""
    ):
        if size_frac <= 0 or size_frac < self.min_size_frac:
            return None
        if not math.isfinite(size_frac):
            raise ValueError(f"Non-finite size_frac {size_frac}")
        if not math.isfinite(market_price):
            raise ValueError(f"Non-finite market_price {market_price}")
        order = {
            "side": "buy" if side > 0 else "sell",
            "size_frac": float(size_frac),
            "price": float(market_price),
            "note": note,
        }
        return order

    def step(
        self,
        model_out: Dict[str, Any],
        market_price: float,
        sl_price_dist: float,
        spread_price: float,
        estimated_slippage_price: float,
        current_bar_index: int,
        extra_inputs: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        model_out: output of model(x_seq)
        extra_inputs can contain: 'kelly_soft_hist','qtp_hist','hazard_hist','forces_hist',
                                 'tail_hist','conf_hist','ev_hist'

        Returns: dict with keys: decision, order (or None), diagnostics
        """

        extra_inputs = extra_inputs or {}

        # ---------- PATCH 1: Conservative execution costs ----------
        spread_price *= 1.25
        estimated_slippage_price *= 1.5

        # ---------- PATCH 2: Tail-based exposure throttle + pause-after-losses ----------
        if self.risk is not None:
            # compute a scalar tail_combined for throttling
            tail_combined_field = model_out["tail"]["tail_combined"]
            if isinstance(tail_combined_field, torch.Tensor):
                tail_combined_val = float(tail_combined_field.mean().item())
            else:
                tail_combined_val = float(tail_combined_field)

            # throttle fmax based on tail
            self.risk.apply_tail_throttle(tail_combined_val)
            self.decision.fmax = self.risk.current_fmax

            # if we're currently flat and in a "pause" window, block new entries
            if (
                self.decision.current_position is None
                and not self.risk.entries_allowed(current_bar_index)
            ):
                return {
                    "decision": {
                        "action": "no_action",
                        "reason": "paused_after_losses",
                        "details": {},
                    },
                    "order": None,
                    "diagnostics": {},
                }

        # ---------- Original logic (unchanged) ----------

        # Explicitly handle tensors and enforce batch size 1 for execution
        ev_tensor = model_out["ev_mean"]
        conf_tensor = model_out["conf"]

        if isinstance(ev_tensor, torch.Tensor):
            if ev_tensor.numel() != 1:
                raise ValueError(
                    f"ExecutionEngine expects ev_mean with a single element (batch size 1); got shape {tuple(ev_tensor.shape)}"
                )
            ev_latest = float(ev_tensor.reshape(-1)[0].item())
        else:
            ev_latest = float(ev_tensor)

        if isinstance(conf_tensor, torch.Tensor):
            if conf_tensor.numel() != 1:
                raise ValueError(
                    f"ExecutionEngine expects conf with a single element (batch size 1); got shape {tuple(conf_tensor.shape)}"
                )
            conf_latest = float(conf_tensor.reshape(-1)[0].item())
        else:
            conf_latest = float(conf_tensor)

        # Build required histories defaulting to simple single-element arrays from model_out
        ev = extra_inputs.get("ev_hist", [ev_latest])
        conf = extra_inputs.get("conf_hist", [conf_latest])

        tail_hist = extra_inputs.get(
            "tail_hist",
            [
                float(
                    model_out["tail"]["tail_combined"].mean()
                    if isinstance(
                        model_out["tail"]["tail_combined"], torch.Tensor
                    )
                    else model_out["tail"]["tail_combined"]
                )
            ],
        )
        kelly_soft = extra_inputs.get("kelly_soft_hist", [0.0])
        qtp = extra_inputs.get("qtp_hist", [0.0])
        hazard_hist = extra_inputs.get("hazard_hist", None)
        forces_hist = extra_inputs.get("forces_hist", None)

        # call decision engine
        decision = self.decision.decide(
            ev_hist=ev,
            conf_hist=conf,
            tail_hist=tail_hist,
            kelly_soft_hist=kelly_soft,
            qtp_hist=qtp,
            hazard_hist=hazard_hist,
            forces_hist=forces_hist,
            spread_price=spread_price,
            estimated_slippage_price=estimated_slippage_price,
            sl_price_dist=sl_price_dist,
            current_bar_index=current_bar_index,
        )

        order = None
        if (
            decision["action"].startswith("enter")
            or decision["action"] in ("add_leg", "partial_exit", "exit")
        ):
            side = decision.get("side", 1)
            size = decision.get("size", 0.0)
            # Convert size fraction to notional or lot size in your adapter.
            order = self._make_order(
                side=side,
                size_frac=size,
                market_price=market_price,
                note=decision.get("reason", ""),
            )

        return {
            "decision": decision,
            "order": order,
            "diagnostics": decision.get("details", {}),
        }

    @torch.no_grad()
    def update_tail_loss(self, realized_pnl: float):
        """
        Accepts signed PnL.
        Negative PnL => loss pushed into tail buffer.
        """
        if realized_pnl >= 0:
            return

        loss = float(-realized_pnl)
        arr = torch.tensor([loss], dtype=torch.float32)
        self.model.tail_module.update_losses(arr)
