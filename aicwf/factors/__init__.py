"""Monthly point-in-time factor construction, neutralisation and diagnostics.

Public stage entrypoint: :func:`aicwf.factors.stage.run_factor_build`.
"""

from __future__ import annotations

__all__ = ["run_factor_build"]


def run_factor_build(ctx):  # pragma: no cover - thin re-export
    from aicwf.factors.stage import run_factor_build as _run

    return _run(ctx)
