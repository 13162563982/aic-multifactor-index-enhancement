"""Generate the holdings-concentration figure from the verified backtest holdings.

Panels:
  1. number of holdings and effective number of names (1 / HHI) over time
  2. top-10 aggregate weight (strategy vs benchmark proxy) and maximum single weight
     against the 3% name cap
  3. latest-date concentration curve: cumulative weight versus rank for the strategy
     and the benchmark proxy

The figure is generated deterministically from the verified `holdings.parquet`; it does
not modify the approved run artifacts.

Usage:
    python scripts/make_concentration_panel.py
"""

from __future__ import annotations

import argparse
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
    ROOT / "docs" / "figures" / "concentration_panel.png",
    ROOT / "examples" / "demo_run" / "10_report" / "charts" / "concentration_panel.png",
]

COLOR_STRATEGY = "#1f4e79"
COLOR_BENCHMARK = "#7f7f7f"
COLOR_CAP = "#c0504d"
COLOR_EFF = "#b26b00"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the holdings-concentration figure")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    holdings_path = Path(args.run_dir) / "05_backtest" / "holdings.parquet"
    if not holdings_path.exists():
        raise SystemExit(f"missing {holdings_path}")
    frame = pd.read_parquet(holdings_path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["weight"] = frame["weight"].astype(float)
    frame["benchmark_weight"] = frame["benchmark_weight"].astype(float)
    frame = frame.sort_values(["date", "weight"], ascending=[True, False])

    def aggregates(group: pd.DataFrame) -> pd.Series:
        weights = group["weight"].to_numpy()
        bench = group["benchmark_weight"].to_numpy()
        return pd.Series(
            {
                "n": int((weights > 0).sum()),
                "invested": float(weights.sum()),
                "max_weight": float(weights.max()),
                "top10": float(np.sort(weights)[::-1][:10].sum()),
                "top10_bench": float(np.sort(bench)[::-1][:10].sum()),
                "hhi": float(np.sum(weights**2)),
            }
        )

    stats = frame.groupby("date").apply(aggregates, include_groups=False)
    stats["eff_n"] = 1.0 / stats["hhi"]
    latest_date = stats.index.max()
    latest = frame[frame["date"] == latest_date]
    latest_sorted = latest.sort_values("weight", ascending=False).reset_index(drop=True)
    bench_sorted = latest["benchmark_weight"].sort_values(ascending=False).reset_index(drop=True)
    metrics_path = Path(args.run_dir) / "05_backtest" / "metrics.json"
    cap = 0.03
    if metrics_path.exists():
        import json

        constraints = json.loads(metrics_path.read_text(encoding="utf-8")).get("constraints") or {}
        cap = float(constraints.get("max_weight") or cap)
    cap_label = f"{cap:.0%} name cap"

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.0, 10.4),
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.35]},
        dpi=args.dpi,
    )
    fig.subplots_adjust(left=0.075, right=0.93, top=0.93, bottom=0.125, hspace=0.32)

    # --- 1. count and effective N ------------------------------------------
    ax = axes[0]
    ax.plot(stats.index, stats["n"], color=COLOR_STRATEGY, linewidth=1.4, label="Number of holdings")
    ax.set_ylabel("number of holdings")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(stats.index, stats["eff_n"], color=COLOR_EFF, linewidth=1.4, label="Effective number (1/HHI)")
    ax2.set_ylabel("effective number of names", color=COLOR_EFF)
    ax2.tick_params(axis="y", colors=COLOR_EFF)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title(
        "Holdings concentration — verified backtest, 2018-02-01 to 2026-11-12",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
    )
    ax.annotate(
        f"latest: {int(stats['n'].iloc[-1])} names, effective N {stats['eff_n'].iloc[-1]:.0f}",
        xy=(0.995, 0.78),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color="#333333",
    )

    # --- 2. top-10 and max weight -----------------------------------------
    ax = axes[1]
    ax.plot(stats.index, stats["top10"], color=COLOR_STRATEGY, linewidth=1.5, label="Top-10 weight (strategy)")
    ax.plot(
        stats.index,
        stats["top10_bench"],
        color=COLOR_BENCHMARK,
        linewidth=1.3,
        linestyle="--",
        label="Top-10 weight (benchmark proxy)",
    )
    ax.plot(stats.index, stats["max_weight"], color=COLOR_CAP, linewidth=1.2, label="Largest single weight")
    ax.axhline(cap, color=COLOR_CAP, linewidth=0.9, linestyle=":", label=cap_label)
    ax.set_ylabel("share of NAV")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8.3, ncol=2)
    ax.annotate(
        f"latest top-10 {stats['top10'].iloc[-1]:.1%} vs benchmark {stats['top10_bench'].iloc[-1]:.1%}; "
        f"max {stats['max_weight'].iloc[-1]:.2%}",
        xy=(0.995, 0.70),
        xycoords="axes fraction",
        ha="right",
        fontsize=8.5,
        color=COLOR_STRATEGY,
    )

    # --- 3. latest-date concentration curve --------------------------------
    ax = axes[2]
    rank = np.arange(1, len(latest_sorted) + 1)
    curve = latest_sorted["weight"].cumsum().to_numpy()
    rank_b = np.arange(1, len(bench_sorted) + 1)
    curve_b = bench_sorted.cumsum().to_numpy()
    ax.plot(rank, curve, color=COLOR_STRATEGY, linewidth=1.6, label="Strategy")
    ax.plot(rank_b, curve_b, color=COLOR_BENCHMARK, linewidth=1.4, linestyle="--", label="Benchmark proxy")
    ax.axvline(10, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("rank of holding (largest first)")
    ax.set_ylabel("cumulative weight")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.annotate(
        f"at rank 10: strategy {curve[9]:.1%}, benchmark {curve_b[9]:.1%}",
        xy=(0.01, 0.93),
        xycoords="axes fraction",
        ha="left",
        fontsize=8.5,
        color="#333333",
    )
    ax.set_title(f"Concentration curve on {latest_date.date()} (information cutoff 2026-11-12)", fontsize=10.0, loc="left")

    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for axis in axes[:2]:
        axis.margins(x=0.01)

    note = (
        "Source: verified holdings.parquet of run final2-2026-11-12 (snapshot 8ab22713…, verification PASS 11/11).\n"
        "Effective number of names = 1 / Herfindahl index.  Invested share ranges "
        f"{stats['invested'].min():.1%}–{stats['invested'].max():.1%} (residual is cash from lot rounding, blocked trades and the initial ramp-up).\n"
        "Benchmark weights are the capped free-float proxy used throughout the backtest; the 3% name cap applies to the strategy only."
    )
    fig.text(0.075, 0.012, note, ha="left", va="bottom", fontsize=7.3, color="#555555", linespacing=1.5)

    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
