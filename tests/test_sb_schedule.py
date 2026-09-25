#!/usr/bin/env python3
"""The public baseline runs the same dt and pump curve as dsb-gpu.

Runs on CPU.  The package-level checks are skipped when torch or
simulated-bifurcation 2.0.0 is not installed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import sb_schedule  # noqa: E402

try:
    import torch  # noqa: F401
    import simulated_bifurcation as sb
    from simulated_bifurcation.core import Ising
    from simulated_bifurcation.optimizer import simulated_bifurcation_optimizer as sbo
    HAVE_SB = getattr(sb, "__version__", None) == "2.0.0"
except ImportError:  # pragma: no cover
    HAVE_SB = False


def dsb_gpu_pump(steps: int):
    """src/solver.cu and src/sparse_solver.cu."""
    if steps == 1:
        return [0.0]
    return [float(i) / float(steps - 1) for i in range(steps)]


def args_for(schedule: str, steps: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sb_schedule.add_arguments(parser)
    args = parser.parse_args(["--sb-schedule=%s" % schedule])
    args.steps = steps
    return args


class SlopeTest(unittest.TestCase):
    def test_formula(self) -> None:
        for steps in (2, 200, 800, 3200):
            slope = sb_schedule.pressure_slope(1.0, steps)
            p = [min(1.0 * k * slope, 1.0) for k in range(steps)]
            for a, b in zip(p, dsb_gpu_pump(steps)):
                self.assertAlmostEqual(a, b, places=12)

    def test_default_is_matched(self) -> None:
        parser = argparse.ArgumentParser()
        sb_schedule.add_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.sb_schedule, "matched")
        self.assertEqual(args.sb_time_step, sb_schedule.DSB_GPU_DT)


@unittest.skipUnless(HAVE_SB, "needs torch and simulated-bifurcation==2.0.0")
class PackageTest(unittest.TestCase):
    """Record the coefficients the package actually uses at every step."""

    def run_package(self, steps: int):
        name = "_SimulatedBifurcationOptimizer__compute_symplectic_coefficients"
        original = getattr(sbo.SimulatedBifurcationOptimizer, name)
        seen = []

        def spy(self_):
            coefficients = original(self_)
            seen.append((coefficients, float(self_.quadratic_scale_parameter)))
            return coefficients

        setattr(sbo.SimulatedBifurcationOptimizer, name, spy)
        try:
            j = torch.tensor([[0.0, -1.0, 1.0], [-1.0, 0.0, -1.0], [1.0, -1.0, 0.0]])
            Ising(j, torch.zeros(3), dtype=torch.float32, device="cpu").minimize(
                agents=4, max_steps=steps, mode="discrete", heated=False,
                verbose=False, early_stopping=False,
            )
        finally:
            setattr(sbo.SimulatedBifurcationOptimizer, name, original)
        return seen

    def tearDown(self) -> None:
        sb.reset_env()
        sb_schedule.set_matched_initial_state(False)

    def test_matched_equals_dsb_gpu(self) -> None:
        for steps in (2, 17, 200, 800):
            self.assertEqual(sb_schedule.configure(sb, args_for("matched", steps)),
                             "discrete-matched")
            seen = self.run_package(steps)
            self.assertEqual(len(seen), steps)
            for ((momentum, position, _), _), p in zip(seen, dsb_gpu_pump(steps)):
                self.assertEqual(position, 1.0)           # dt
                self.assertAlmostEqual(momentum, p - 1.0, places=12)  # dt*(p-1)
            self.assertAlmostEqual(seen[-1][0][0], 0.0, places=12)    # p ends at 1

    def test_matched_initial_state(self) -> None:
        from simulated_bifurcation.optimizer.symplectic_integrator import (
            SymplecticIntegrator,
        )
        sb_schedule.configure(sb, args_for("matched", 10))
        si = SymplecticIntegrator((500, 64), torch.sign, torch.float32, "cpu")
        for t in (si.position, si.momentum):
            self.assertLessEqual(float(t.abs().max()), 0.01)
            self.assertGreater(float(t.abs().max()), 0.009)
        sb_schedule.configure(sb, args_for("library", 10))
        si = SymplecticIntegrator((500, 64), torch.sign, torch.float32, "cpu")
        self.assertGreater(float(si.position.abs().max()), 0.9)

    def test_library_restores_defaults(self) -> None:
        sb_schedule.configure(sb, args_for("matched", 50))
        self.assertEqual(sb_schedule.configure(sb, args_for("library", 800)),
                         "discrete")
        self.assertEqual(sb.get_env()["time_step"], 0.1)
        self.assertEqual(sb.get_env()["pressure_slope"], 0.01)
        seen = self.run_package(800)
        # p = min(k/1000, 1): still 0.799 on the last of 800 steps.
        self.assertAlmostEqual(seen[-1][0][0], 0.1 * (0.799 - 1.0), places=9)


if __name__ == "__main__":
    unittest.main()
