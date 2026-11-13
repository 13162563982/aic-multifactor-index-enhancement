"""Independent verification subsystem.

Public entry points:
    aicwf.verify.stage.run_verification(ctx)      -- pipeline stage handler
    aicwf.verify.stage.build_verification(...)    -- verify a run directory offline
    aicwf.verify.stage.is_blocking(verification)  -- FAIL -> bounded revision
    aicwf.verify.stage.render_verifier_note(v)    -- deterministic Markdown fallback

The recomputation modules (:mod:`independent_backtest`, :mod:`reconcile`) do not
import ``aicwf.backtest``/``aicwf.portfolio``/``aicwf.factors``.
"""

from aicwf.verify.stage import (  # noqa: F401 - re-exported public surface
    build_verification,
    generate_verifier_note_text,
    is_blocking,
    render_verifier_note,
    run_verification,
    verify_error_summary,
)

__all__ = [
    "build_verification",
    "generate_verifier_note_text",
    "is_blocking",
    "render_verifier_note",
    "run_verification",
    "verify_error_summary",
]
