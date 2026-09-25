"""Read reductions exported by ``export_fasthare`` and lift torch spins."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib

import numpy as np


@dataclass
class FastHareReduction:
    kind: str
    name: str
    n_original: int
    n_standard: int
    n_reduced: int
    fully_reduced: bool
    alpha: float
    preprocess_s: float
    sign: np.ndarray
    spin_map: np.ndarray
    coupling: np.ndarray

    @property
    def reduction_ratio(self) -> float:
        return 1.0 - self.n_reduced / self.n_standard


def _label(tokens: list[str], expected: str) -> list[str]:
    if not tokens or tokens[0] != expected:
        raise ValueError("expected %s, found %s" % (expected, tokens[:1]))
    return tokens[1:]


def load_reduction(path: pathlib.Path, expected_kind: str) -> FastHareReduction:
    with path.open("r", encoding="utf-8") as stream:
        if stream.readline().strip() != "DSB_FASTHARE_V1":
            raise ValueError("unsupported FastHare export: %s" % path)
        kind = _label(stream.readline().split(), "kind")[0]
        name = _label(stream.readline().split(), "name")[0]
        n_original = int(_label(stream.readline().split(), "n_original")[0])
        n_standard = int(_label(stream.readline().split(), "n_standard")[0])
        n_reduced = int(_label(stream.readline().split(), "n_reduced")[0])
        fully_reduced = bool(
            int(_label(stream.readline().split(), "fully_reduced")[0])
        )
        alpha = float(_label(stream.readline().split(), "alpha")[0])
        preprocess_s = float(
            _label(stream.readline().split(), "preprocess_s")[0]
        )
        sign_count = int(_label(stream.readline().split(), "sign_count")[0])
        sign = np.asarray(
            [int(value) for value in _label(stream.readline().split(), "sign")],
            dtype=np.int8,
        )
        map_count = int(_label(stream.readline().split(), "map_count")[0])
        spin_map = np.asarray(
            [int(value) for value in _label(stream.readline().split(), "map")],
            dtype=np.int64,
        )
        edge_count = int(_label(stream.readline().split(), "edge_count")[0])
        coupling = np.zeros((n_reduced, n_reduced), dtype=np.float32)
        for edge_index in range(edge_count):
            fields = _label(stream.readline().split(), "edge")
            if len(fields) != 3:
                raise ValueError("invalid edge %d in %s" % (edge_index, path))
            row, column = int(fields[0]), int(fields[1])
            value = float(fields[2])
            if not (0 <= row < n_reduced and 0 <= column < n_reduced):
                raise ValueError("FastHare edge index out of range")
            if row == column or not np.isfinite(value):
                raise ValueError("invalid FastHare edge")
            coupling[row, column] = value
            coupling[column, row] = value
        if stream.read().strip():
            raise ValueError("extra data after FastHare edges in %s" % path)

    if kind != expected_kind:
        raise ValueError("expected %s reduction, found %s" % (expected_kind, kind))
    if sign.size != sign_count or spin_map.size != map_count:
        raise ValueError("FastHare map/sign count mismatch in %s" % path)
    # QPLIB standard form has one extra gauge/bias node.  When the original
    # instance has no linear field, that node is isolated and FastHare omits it
    # from both the sign and map arrays.  Accept exactly that one-node omission;
    # all other size mismatches remain errors.
    qplib_without_gauge = (
        kind == "qplib"
        and n_standard == n_original + 1
        and sign.size == n_original
    )
    if sign.size != n_standard and not qplib_without_gauge:
        raise ValueError("FastHare sign length does not match standard size")
    if not np.all(np.isin(sign, (-1, 1))):
        raise ValueError("FastHare signs must be -1 or +1")
    expected_map_size = sign.size
    if not fully_reduced and spin_map.size != expected_map_size:
        raise ValueError("FastHare map length does not match standard size")
    if fully_reduced and n_reduced != 0:
        raise ValueError("fully reduced export has nonzero residual size")
    if not fully_reduced and (
        np.any(spin_map < 0) or np.any(spin_map >= n_reduced)
    ):
        raise ValueError("FastHare map index out of range")
    return FastHareReduction(
        kind, name, n_original, n_standard, n_reduced, fully_reduced,
        alpha, preprocess_s, sign, spin_map, coupling
    )


def lift_spins(torch, reduced_spins, reduction: FastHareReduction, device: str):
    """Return original-variable spins for every agent on the same device."""
    dtype = reduced_spins.dtype
    sign = torch.as_tensor(reduction.sign, dtype=dtype, device=device)[:, None]
    if reduction.fully_reduced:
        agents = reduced_spins.shape[1]
        full = sign.expand(-1, agents)
    else:
        mapping = torch.as_tensor(
            reduction.spin_map, dtype=torch.long, device=device
        )
        full = reduced_spins[mapping, :] * sign
    if reduction.kind == "qplib":
        # A zero-field QPLIB instance has no gauge node in FastHare's output.
        # In that case the omitted, completely free gauge can be fixed to +1.
        if full.shape[0] > reduction.n_original:
            gauge = full[reduction.n_original, :]
            return full[: reduction.n_original, :] * gauge[None, :]
        return full[: reduction.n_original, :]
    return full[: reduction.n_original, :]
