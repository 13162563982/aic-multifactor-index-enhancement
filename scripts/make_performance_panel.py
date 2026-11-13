"""Generate the four-panel performance figure used in the README and design note.

Panels (shared date axis):
  1. strategy NAV and CSI 300 benchmark NAV (both rebased to 1.0 at inception)
  2. strategy and benchmark drawdown
  3. cumulative excess return (active NAV - 1)
  4. rolling annualised tracking error (63-day and 252-day windows) with the
     ex-ante target and the monitoring alert threshold

The figure is generated deterministically from the verified `equity_curve.csv` and
`metrics.json` of a run. It does not modify the run artifacts, so the approved
report package keeps its original artifacts and hashes.

Usage:
    python scripts/make_performance_panel.py
    python scripts/make_performance_panel.py --run-dir outputs/runs/final2-2026-11-12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs" / "runs" / "final2-2026-11-12"
OUTPUTS = [
    ROOT / "docs" / "figures" / "performance_panel.png",
    ROOT / "examples" / "demo_run" / "10_report" / "charts" / "performance_panel.png",
]

COLOR_STRATEGY = "#1f4e79"
COLOR_BENCHMARK = "#7f7f7f"
COLOR_DRAWDOWN = "#c0504d"
COLOR_EXCESS = "#2e7d32"
COLOR_TE = "#b26b00"
COLOR_TARGET = "#1f4e79"
COLOR_ALERT = "#c0504d"


def _rolling_annualised_te(excess: pd.Series, window: int) -> pd.Series:
    return excess.rolling(window, min_periods=max(10, window // 3)).std() * np.sqrt(252.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the four-panel performance figure")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    equity_path = run_dir / "05_backtest" / "equity_curve.csv"
    metrics_path = run_dir / "05_backtest" / "metrics.json"
    if not equity_path.exists():
        raise SystemExit(f"missing {equity_path}")
    frame = pd.read_csv(equity_path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = doc["metrics"]
    constraints = doc.get("constraints") or {}
    te_target = float(constraints.get("te_target") or 0.03)
    te_alert = 0.08
    realized_te = float(metrics.get("tracking_error", np.nan))
    max_dd = float(metrics.get("max_drawdown", np.nan))
    bench_max_dd = float(metrics.get("benchmark_max_drawdown", np.nan))

    nav = frame["nav"].astype(float)
    benchmark = frame["benchmark_nav"].astype(float)
    nav_plot = nav / nav.iloc[0]
    bench_plot = benchmark / benchmark.iloc[0]
    drawdown = frame["drawdown"].astype(float)
    bench_drawdown = benchmark / benchmark.cummax() - 1.0
    active = frame["active_nav"].astype(float) - 1.0
    excess_daily = frame["excess_return"].astype(float)
    te_63 = _rolling_annualised_te(excess_daily, 63)
    te_252 = _rolling_annualised_te(excess_daily, 252)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12.0, 11.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.15, 1.15, 1.15]},
        dpi=args.dpi,
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.955, bottom=0.065, hspace=0.22)

    # --- 1. NAV -------------------------------------------------------------
    ax = axes[0]
    ax.plot(frame["date"], nav_plot, color=COLOR_STRATEGY, linewidth=1.6, label="Strategy NAV")
    ax.plot(frame["date"], bench_plot, color=COLOR_BENCHMARK, linewidth=1.4, label="CSI 300 benchmark NAV")
    ax.fill_between(
        frame["date"], bench_plot, nav_plot, where=nav_plot >= bench_plot,
        color=COLOR_STRATEGY, alpha=0.08, interpolate=True,
    )
    ax.set_ylabel("NAV (rebased = 1.0)")
    ax.set_title(
        "CSI 300 index enhancement — verified backtest, 2018-02-01 to 2026-11-12\n"
        f"strategy NAV {nav_plot.iloc[-1]:.2f}×   ·   CSI 300 {bench_plot.iloc[-1]:.2f}×   ·   "
        f"annualised excess {metrics.get('annualized_excess_return', 0):+.2%}   ·   "
        f"IR {metrics.get('information_ratio', 0):.2f}   ·   max drawdown {max_dd:.1%}",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)

    # --- 2. drawdown --------------------------------------------------------
    ax = axes[1]
    ax.fill_between(frame["date"], drawdown, 0.0, color=COLOR_DRAWDOWN, alpha=0.35, label="Strategy")
    ax.plot(frame["date"], drawdown, color="#7f1d1d", linewidth=0.9)
    ax.plot(frame["date"], bench_drawdown, color=COLOR_BENCHMARK, linewidth=0.9, label="CSI 300")
    ax.set_ylabel("drawdown")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=8.5)
    ax.annotate(
        f"max {max_dd:.1%}  (benchmark {bench_max_dd:.1%})",
        xy=(0.995, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color="#7f1d1d",
    )

    # --- 3. cumulative excess ----------------------------------------------
    ax = axes[2]
    ax.plot(frame["date"], active, color=COLOR_EXCESS, linewidth=1.5, label="Cumulative excess (NAV ratio − 1)")
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.fill_between(frame["date"], 0.0, active, where=active >= 0, color=COLOR_EXCESS, alpha=0.10)
    ax.set_ylabel("excess vs benchmark")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.annotate(
        f"end {active.iloc[-1]:+.1%}   ·   annualised excess {metrics.get('annualized_excess_return', 0):+.2%}"
        f"   ·   IR {metrics.get('information_ratio', 0):.2f}   ·   TE {realized_te:.2%}",
        xy=(0.995, 0.07),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color=COLOR_EXCESS,
    )

    # --- 4. tracking error --------------------------------------------------
    ax = axes[3]
    ax.plot(frame["date"], te_63, color=COLOR_TE, linewidth=1.3, label="Rolling 63-day annualised TE")
    ax.plot(frame["date"], te_252, color="#5b3a00", linewidth=1.5, label="Rolling 252-day annualised TE")
    ax.axhline(te_target, color=COLOR_TARGET, linewidth=1.0, linestyle="--", label=f"Ex-ante target {te_target:.1%}")
    ax.axhline(te_alert, color=COLOR_ALERT, linewidth=1.0, linestyle=":", label=f"Monitoring alert {te_alert:.0%}")
    ax.set_ylabel("annualised TE")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for axis in axes:
        axis.margins(x=0.01)

    note = (
        "Source: verified equity_curve.csv + metrics.json of run final2-2026-11-12 (snapshot 8ab22713…, verification PASS 11/11). "
        "Tracking error = annualised rolling standard deviation of daily excess returns.\n"
        "Universe = current CSI 300 membership; survivorship and reconstitution bias disclosed in the research note. "
        "Full-sample TE is annotated in the third panel."
    )
    fig.text(0.075, 0.012, note, ha="left", va="bottom", fontsize=7.4, color="#555555", linespacing=1.5)

    written = []
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        written.append(path)
    plt.close(fig)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
