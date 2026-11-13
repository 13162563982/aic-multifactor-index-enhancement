"""Monitoring subsystem: monthly performance, drift and data-freshness report.

Public entry points:
    aicwf.monitor.run.run_monitor(as_of, llm_mode, monitor_id=None, ...) -> dict
    aicwf.monitor.run.main(argv) -> int      -- ``python -m aicwf.monitor`` CLI
    aicwf.monitor.drift.*                    -- pure analytics helpers

Outputs land in ``outputs/monitor/<monitor_id>/`` per docs/artifact_contract.md.
"""

from aicwf.monitor.drift import (  # noqa: F401
    coverage,
    data_freshness,
    detect_new_data,
    exposure_drift,
    factor_rank_ic,
    mark_to_market_performance,
)
from aicwf.monitor.run import (  # noqa: F401
    build_alerts,
    monitoring_thresholds,
    render_monitoring_report,
    run_monitor,
)

__all__ = [
    "build_alerts",
    "coverage",
    "data_freshness",
    "detect_new_data",
    "exposure_drift",
    "factor_rank_ic",
    "mark_to_market_performance",
    "monitoring_thresholds",
    "render_monitoring_report",
    "run_monitor",
]
