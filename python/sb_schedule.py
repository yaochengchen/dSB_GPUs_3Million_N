"""Integration schedule of the public simulated-bifurcation 2.0.0 baseline.

The package keeps its hyper-parameters in a process-global environment
(``sb.set_env``), defaulting to ``time_step=0.1, pressure_slope=0.01``.  Its
pump is

    p_k = min(time_step * k * pressure_slope, 1),   k = 0 .. max_steps-1

which does not depend on ``max_steps``: with the defaults p reaches 1 at step
1000, and the library expects to run on (``max_steps=10_000``, early stopping)
at p = 1 until the agents converge.  Run with a fixed ``max_steps=800`` it is
cut off at p = 0.8 (and at p = 0.2 for 200 steps) -- a different annealing
curve from the one dsb-gpu runs.

dsb-gpu (``Options`` in include/dsb/solver.hpp) uses ``dt = 1``, ``delta = 1``
and ``p_k = k / (n_steps - 1)``, i.e. p reaches exactly 1 on the last step for
every step count.  The ``matched`` schedule sets the package to the same curve:

    time_step      = dt                        (1.0)
    pressure_slope = 1 / (dt * (steps - 1))    -> p_k = k / (steps - 1)

    x_0, y_0       ~ uniform(-0.01, 0.01)      (package: uniform(-1, 1))

The initial state is set by replacing the package's (private) oscillator
initialiser for the duration of the process; the draws still come from torch's
global RNG, which the benchmark seeds per repeat, so runs stay reproducible.
The distribution matches dsb-gpu, the individual numbers cannot (dsb-gpu uses
its own RNG).

xi (``quadratic_scale_parameter``), the detuning (1), the inelastic walls and
the sign activation already coincide; the only remaining difference is the
order of the updates inside one step (the package applies the coupling term
after the position update, dsb-gpu before).

``library`` restores the package defaults and is kept for the "as published"
comparison; its rows are labelled ``discrete`` so old results stay comparable.
Matched rows are labelled ``discrete-matched`` so the summaries never merge the
two.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

SCHEDULES = ("matched", "library")

# Must equal dsb::Options::dt in include/dsb/solver.hpp.
DSB_GPU_DT = 1.0
# dsb-gpu initial state: x, y ~ 0.02 * (U(0,1) - 0.5)  (src/solver.cu, reference_dsb.py)
DSB_GPU_INIT_HALF_WIDTH = 0.01

_INIT_ATTR = "_SymplecticIntegrator__init_oscillator"
_original_init: Optional[Any] = None


def _integrator_class():
    from simulated_bifurcation.optimizer import symplectic_integrator
    return symplectic_integrator.SymplecticIntegrator


def set_matched_initial_state(enabled: bool) -> None:
    """Swap the package's uniform(-1,1) oscillator init for dsb-gpu's."""
    global _original_init
    cls = _integrator_class()
    if _original_init is None:
        _original_init = cls.__dict__[_INIT_ATTR]  # the staticmethod object
    if not enabled:
        setattr(cls, _INIT_ATTR, _original_init)
        return

    def init_oscillator(shape, dtype, device):
        import torch
        half = DSB_GPU_INIT_HALF_WIDTH
        return 2.0 * half * (torch.rand(size=shape, device=device, dtype=dtype) - 0.5)

    setattr(cls, _INIT_ATTR, staticmethod(init_oscillator))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sb-schedule", choices=SCHEDULES, default="matched",
        help="matched: dt, pump curve and initial state of dsb-gpu (default); "
             "library: simulated-bifurcation 2.0.0 defaults "
             "(time_step=0.1, pressure_slope=0.01)",
    )
    parser.add_argument(
        "--sb-time-step", type=float, default=DSB_GPU_DT,
        help="time step used by --sb-schedule=matched; keep equal to the "
             "dt of dsb-gpu (default %(default)s)",
    )


def pressure_slope(time_step: float, steps: int) -> float:
    """Slope that makes p_k = k / (steps - 1) under the package's formula."""
    if steps <= 1:
        return 0.0  # dsb-gpu uses p_0 = 0 for a single step
    return 1.0 / (float(time_step) * float(steps - 1))


def configure(sb: Any, args: argparse.Namespace) -> str:
    """Set the package environment for this run; return the variant label."""
    sb.reset_env()
    if args.sb_schedule == "library":
        set_matched_initial_state(False)
        return "discrete"
    set_matched_initial_state(True)
    if not args.sb_time_step > 0.0:
        raise ValueError("--sb-time-step must be positive")
    sb.set_env(
        time_step=float(args.sb_time_step),
        pressure_slope=pressure_slope(args.sb_time_step, args.steps),
    )
    return "discrete-matched"


def describe(sb: Any) -> Dict[str, Any]:
    env: Dict[str, Any] = dict(sb.get_env())
    matched = _integrator_class().__dict__[_INIT_ATTR] is not _original_init
    env["init"] = ("uniform(-%g,%g)" % ((DSB_GPU_INIT_HALF_WIDTH,) * 2)
                   if matched else "uniform(-1,1)")
    return env
