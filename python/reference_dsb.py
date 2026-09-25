"""PyTorch reference implementation of discrete Simulated Bifurcation (dSB).

This is the baseline the CUDA kernels are measured against. It is deliberately
plain: one ``torch.sparse.mm`` plus a handful of elementwise kernels per step,
no fusion.

Two things that were wrong in the earlier version and that matter if you are
going to quote a speedup:

* the pump schedule now lives on the same device as ``x``. It used to be a CPU
  tensor, so every step paid an implicit host-to-device transfer and a sync --
  800 syncs over a run, which inflates any speedup measured against it;
* ``to_sparse_csr()`` is done once in the constructor, not inside the loop.

Reference:
    H. Goto et al., "High-performance combinatorial optimization based on
    classical mechanics", Sci. Adv. 7, eabe7953 (2021).
"""

from __future__ import annotations

import math

import torch

__all__ = ["DiscreteSB"]


class DiscreteSB:
    """Discrete Simulated Bifurcation over a sparse coupling matrix.

    Args:
        coupling: (N, N) coupling matrix, dense or sparse.
        field: optional (N,) or (N, 1) external field. When given it is folded
            into the coupling matrix as one extra spin, the same way the C++
            pipeline does it.
        n_steps: integration steps.
        batch: independent replicas.
        dt: step size.
        xi: coupling gain. ``None`` selects 0.5*sqrt(N-1)/||J||_F.
        device: torch device.
        seed: RNG seed, or ``None`` for whatever torch's global state holds.
    """

    def __init__(
        self,
        coupling,
        field=None,
        n_steps: int = 1000,
        batch: int = 1,
        dt: float = 1.0,
        xi: float | None = None,
        device: str = "cuda",
        seed: int | None = 12345,
    ):
        self.device = torch.device(device)
        self.n_steps = int(n_steps)
        self.batch = int(batch)
        self.dt = float(dt)
        self.delta = 1.0

        coupling = coupling.to(self.device)
        if field is not None:
            field = field.to(self.device)
            if field.dim() == 1:
                field = field.view(-1, 1)
            coupling = self._absorb_field(coupling, field)
        self.field = field

        if coupling.layout is not torch.sparse_csr:
            coupling = coupling.to_sparse_csr()
        self.coupling = coupling
        self.n = self.coupling.shape[0]

        if xi is None:
            frobenius = torch.sqrt((self.coupling.to_dense() ** 2).sum())
            xi = 0.5 * math.sqrt(self.n - 1) / frobenius.item()
        self.xi = float(xi)

        # On device, so the step loop never touches the host.
        self.pump = torch.linspace(0.0, 1.0, self.n_steps, device=self.device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        self.x = 0.02 * (
            torch.rand(self.n, self.batch, device=self.device, generator=generator) - 0.5
        )
        self.y = 0.02 * (
            torch.rand(self.n, self.batch, device=self.device, generator=generator) - 0.5
        )

    @staticmethod
    def _absorb_field(coupling, field):
        """Fold the external field in as one extra spin at index N."""
        n = coupling.shape[0]
        dense = coupling.to_dense() if coupling.layout is not torch.strided else coupling
        dense = (dense + dense.t()) / 2.0

        out = torch.zeros((n + 1, n + 1), device=dense.device, dtype=dense.dtype)
        out[:n, :n] = dense
        out[:n, n] = -field[:, 0]
        out[n, :n] = -field[:, 0]
        return out

    def run(self):
        """Integrate all steps. Modifies ``self.x`` and ``self.y`` in place."""
        for i in range(self.n_steps):
            coupling_term = torch.sparse.mm(self.coupling, torch.sign(self.x))
            self.y += (
                -(self.delta - self.pump[i]) * self.x + self.xi * coupling_term
            ) * self.dt
            self.x += self.dt * self.y * self.delta

            clipped = torch.abs(self.x) > 1.0
            self.x = torch.where(clipped, torch.sign(self.x), self.x)
            self.y = torch.where(clipped, torch.zeros_like(self.y), self.y)
        return self.x

    def energies(self, x=None):
        """Ising energy per replica: -0.5 * sum_i (J @ s)_i * s_i."""
        s = torch.sign(self.x if x is None else x)
        return -0.5 * torch.sum(torch.sparse.mm(self.coupling, s) * s, dim=0)

    def cuts(self, x=None):
        """Max-cut value per replica."""
        s = torch.sign(self.x if x is None else x)
        return 0.25 * torch.sum(
            torch.sparse.mm(self.coupling, s) * s, dim=0
        ) - 0.25 * self.coupling.to_dense().sum()

    def best(self):
        """(index, spins) of the lowest-energy replica."""
        e = self.energies()
        i = int(torch.argmin(e).item())
        return i, torch.sign(self.x[:, i])
