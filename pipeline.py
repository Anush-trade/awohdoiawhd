from typing import List, Dict, Any, Optional, Sequence, Mapping
import os
import contextlib

import numpy as np
import torch

from preprocessors.ofi_preprocessor_v2 import OFIPreprocessorV2
from preprocessors.liquidity_preprocessor_v2 import LiquidityHuntPreprocessorV2
from preprocessors.volshock_preprocessor_v2 import VolShockPreprocessorV2
from preprocessors.trend_preprocessor_v2 import TrendPreprocessorV2
from preprocessors.meanrev_preprocessor_v2 import MeanRevertPreprocessorV2
from preprocessors.inventory_preprocessor_v2 import InventoryPressurePreprocessorV2

from wrappers.ofi_wrapper_v2 import OFIWrapperV2
from wrappers.liquidity_wrapper_v2 import LiquidityForceWrapperV2
from wrappers.volshock_wrapper_v2 import VolShockWrapperV2
from wrappers.trend_wrapper_v2 import TrendForceWrapperV2
from wrappers.meanrev_wrapper_v2 import MeanRevWrapperV2
from wrappers.inventory_wrapper_v2 import InventoryForceWrapperV2


def load_pca_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PCA parameters from a .npz file.

    Expected keys:
        - 'mu': mean vector, shape (D,)
        - 'W' : projection matrix, shape (D, K)

    Returns
    -------
    mu : np.ndarray
        Mean vector as float32.
    W : np.ndarray
        Projection matrix as float32.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PCA file not found at: {path}")

    arr = np.load(path, allow_pickle=False)

    if "mu" not in arr or "W" not in arr:
        raise KeyError(
            f"PCA file {path} must contain 'mu' and 'W' arrays. "
            f"Found keys: {list(arr.keys())}"
        )

    # Use float32 explicitly for consistency/performance.
    mu = arr["mu"].astype(np.float32)
    W = arr["W"].astype(np.float32)

    if mu.ndim != 1:
        raise ValueError(f"PCA 'mu' in {path} must be 1D, got shape {mu.shape}")
    if W.ndim != 2:
        raise ValueError(f"PCA 'W' in {path} must be 2D, got shape {W.shape}")
    if W.shape[0] != mu.shape[0]:
        raise ValueError(
            f"PCA dimensions mismatch in {path}: "
            f"mu.shape={mu.shape}, W.shape={W.shape}"
        )

    return mu, W


class PreprocessingPipeline:
    """
    Composes multiple force-specific preprocessors + wrappers into a single API.

    Forces (by default):
        - 'ofi'
        - 'liquidity'
        - 'volshock'
        - 'trend'
        - 'meanrev'
        - 'inventory'

    After preprocessing + wrapping, returns a tensor of shape:
        (B, 1, num_forces, per_force_in)

    Parameters
    ----------
    pca_map : Dict[str, str]
        Mapping from force name -> path to PCA .npz file.
    per_force_in : int, default=32
        Output dimension per force after wrapper.
    device : str or torch.device, default="cpu"
        Device where wrappers and output tensors will live.
    wrappers_map : Optional[Dict[str, torch.nn.Module]]
        Optional override for force wrappers: {force_name: nn.Module}.
    forces : Optional[Sequence[str]]
        Subset/ordering of forces to use. If None, DEFAULT_FORCES is used.
    inference : bool, default=True
        If True, wrappers are put into eval() mode and compute_forces_batch
        runs under torch.no_grad(). Set False if you want gradients.
    """

    DEFAULT_FORCES: Sequence[str] = (
        "ofi",
        "liquidity",
        "volshock",
        "trend",
        "meanrev",
        "inventory",
    )

    def __init__(
        self,
        pca_map: Mapping[str, str],
        per_force_in: int = 32,
        device: str | torch.device = "cpu",
        wrappers_map: Optional[Dict[str, torch.nn.Module]] = None,
        forces: Optional[Sequence[str]] = None,
        inference: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.per_force_in = int(per_force_in)
        self.inference = bool(inference)

        # Determine which forces to use
        self.forces: Sequence[str] = list(forces) if forces is not None else list(
            self.DEFAULT_FORCES
        )

        if not self.forces:
            raise ValueError("Forces list cannot be empty.")

        # Validate PCA map covers all requested forces
        missing = [f for f in self.forces if f not in pca_map]
        if missing:
            raise KeyError(
                f"PCA map is missing entries for forces: {missing}. "
                f"Provided keys: {list(pca_map.keys())}"
            )

        self.pca_map: Dict[str, str] = dict(pca_map)
        self.preprocs: Dict[str, Any] = {}
        self.wrappers: Dict[str, torch.nn.Module] = {}
        # Expected input feature dimension per force (K)
        self.force_input_dims: Dict[str, int] = {}

        self._build_preprocessors(wrappers_override=wrappers_map or {})

    def _build_preprocessors(self, wrappers_override: Dict[str, torch.nn.Module]) -> None:
        """
        Build preprocessors and corresponding wrappers for each force.
        Injects frozen PCA parameters into preprocessors.
        Validates wrapper IO shapes at construction time.
        """

        registry: Dict[str, tuple[type, type]] = {
            "ofi": (OFIPreprocessorV2, OFIWrapperV2),
            "liquidity": (LiquidityHuntPreprocessorV2, LiquidityForceWrapperV2),
            "volshock": (VolShockPreprocessorV2, VolShockWrapperV2),
            "trend": (TrendPreprocessorV2, TrendForceWrapperV2),
            "meanrev": (MeanRevertPreprocessorV2, MeanRevWrapperV2),
            "inventory": (InventoryPressurePreprocessorV2, InventoryForceWrapperV2),
        }

        for fname in self.forces:
            if fname not in registry:
                raise KeyError(f"Unknown force name '{fname}' (not in registry).")

            PreprocClass, WrapperClass = registry[fname]

            # Load PCA (mu, W) for this force
            mu, W = load_pca_npz(self.pca_map[fname])
            in_k = int(W.shape[1])

            # Build preprocessor with frozen PCA
            pre = PreprocClass(pca_mu=mu, pca_W=W)

            # Build / override wrapper
            wrap = wrappers_override.get(fname)
            if wrap is None:
                wrap = WrapperClass(in_k=in_k, per_force_in=self.per_force_in)

            if not isinstance(wrap, torch.nn.Module):
                raise TypeError(
                    f"Wrapper for force '{fname}' must be an nn.Module, "
                    f"got type {type(wrap)}"
                )

            wrap = wrap.to(self.device)

            if self.inference:
                wrap.eval()

            # Validate wrapper IO with a dry run
            with torch.no_grad():
                dummy = torch.zeros(2, 1, in_k, device=self.device, dtype=torch.float32)
                out = wrap(dummy)

            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    f"Wrapper for '{fname}' must return a torch.Tensor, "
                    f"got {type(out)}"
                )

            if out.ndim != 3 or out.shape[0] != 2 or out.shape[1] != 1 or out.shape[2] != self.per_force_in:
                raise ValueError(
                    f"Wrapper for '{fname}' must return shape (B,1,{self.per_force_in}), "
                    f"got {tuple(out.shape)}"
                )

            # Record components
            self.preprocs[fname] = pre
            self.wrappers[fname] = wrap
            self.force_input_dims[fname] = in_k

    def compute_forces_batch(
        self,
        batch_inputs: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Run all configured forces on a batch of inputs.

        Parameters
        ----------
        batch_inputs : List[Dict[str, Any]]
            Each element must contain:
                - 'current_raw': raw current snapshot
                - optional 'history_raws': list of past snapshots

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, 1, num_forces, per_force_in)
            in the pipeline's device and float32 dtype.
        """
        B = len(batch_inputs)

        # Allow empty batches: return a correctly-shaped empty tensor.
        if B == 0:
            return torch.empty(
                0,
                1,
                len(self.forces),
                self.per_force_in,
                device=self.device,
                dtype=torch.float32,
            )

        # Basic input validation
        for i, sample in enumerate(batch_inputs):
            if "current_raw" not in sample:
                raise KeyError(f"batch_inputs[{i}] is missing 'current_raw'")

        per_force_tensors: Dict[str, torch.Tensor] = {}

        # Still per-sample loops since preprocessors likely expect (current, history)
        # as Python objects; batching them would require changing preprocessor APIs.
        for fname in self.forces:
            pre = self.preprocs[fname]
            expected_k = self.force_input_dims[fname]
            out_list: list[np.ndarray] = []

            for sample in batch_inputs:
                current = sample["current_raw"]
                # Make sure we don't share mutable history lists between calls.
                history_raws = sample.get("history_raws", [])
                history = list(history_raws) if history_raws is not None else []

                feat32, _meta = pre.compute(current, history)

                if not isinstance(feat32, np.ndarray):
                    feat32 = np.asarray(feat32, dtype=np.float32)
                else:
                    feat32 = feat32.astype(np.float32, copy=False)

                # ensure feature dimension matches expectation (K)
                if feat32.ndim != 1:
                    raise ValueError(
                        f"Preprocessor for '{fname}' must return 1D feature vector, "
                        f"got shape {feat32.shape}"
                    )
                if feat32.shape[0] != expected_k:
                    raise ValueError(
                        f"Feature dimension mismatch for '{fname}': "
                        f"expected {expected_k}, got {feat32.shape[0]}"
                    )

                out_list.append(feat32)

            # (B, K); ensure contiguous float32 for torch.from_numpy
            arr = np.stack(out_list, axis=0)
            arr = np.ascontiguousarray(arr, dtype=np.float32)

            t = torch.from_numpy(arr).to(device=self.device)  # (B, K)
            t = t.unsqueeze(1)  # (B, 1, K)
            per_force_tensors[fname] = t

        wrapped_outputs: list[torch.Tensor] = []

        ctx = torch.no_grad() if self.inference else contextlib.nullcontext()
        with ctx:
            for fname in self.forces:
                w = self.wrappers[fname]
                x = per_force_tensors[fname]  # (B, 1, K)
                y = w(x)                      # (B, 1, per_force_in)

                if not isinstance(y, torch.Tensor):
                    raise TypeError(
                        f"Wrapper for '{fname}' must return a torch.Tensor, "
                        f"got {type(y)}"
                    )

                if y.ndim != 3 or y.shape[1] != 1 or y.shape[2] != self.per_force_in:
                    raise ValueError(
                        f"Wrapper for '{fname}' must return (B,1,{self.per_force_in}), "
                        f"got shape {tuple(y.shape)}"
                    )

                # Enforce device & dtype consistency in case wrapper created new tensors.
                y = y.to(self.device, dtype=torch.float32)
                wrapped_outputs.append(y)

        if not wrapped_outputs:
            # Should be impossible because we disallow empty forces,
            # but keep this as a safety net.
            raise RuntimeError("No wrapped outputs were produced (no forces?)")

        # Stack along force dimension -> (B, 1, num_forces, per_force_in)
        stacked = torch.stack(wrapped_outputs, dim=2)
        return stacked
