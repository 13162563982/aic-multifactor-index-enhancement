# Finance methodology

This note documents the quantitative definitions behind the workflow. It is the reference for
`aicwf/factors`, `aicwf/backtest`, `aicwf/portfolio` and the independent verifier. All formulas
are implemented in deterministic code; none of them is delegated to a language model.

## 1. Universe and benchmark

The benchmark is the CSI 300 price index (`000300.SH`). The investable universe is the current
CSI 300 constituent list retrieved from the public Eastmoney board endpoint `b:BK0500`
(300 names) and frozen in the snapshot. Benchmark weights are approximated by free-float market
capitalisation within the constituent set, normalised to one. The published CSI 300 weighting
scheme applies a 10% cap and a tiered free-float adjustment; the proxy is close but not
identical, which is documented as a limitation and validated in the robustness section.

This is a **current-constituent** snapshot, so the backtest contains survivorship and
reconstitution bias. The workflow does not hide it: the robustness stage runs an adversarial
stress test that removes the 30 names with the best full-period returns and re-runs the
strategy, and the design note and research note disclose the bias. The result is a research
prototype, not a production performance claim.

Filters applied before a name can enter a portfolio:

* listed for at least 250 trading days before the decision date;
* not flagged ST/*ST;
* not suspended on the decision or execution date;
* at least 90% non-missing price observations while a member;
* eligible price (positive, not stale beyond the configured limit).

## 2. Point-in-time data rules

* Prices are forward-adjusted (`fqt=1`) daily bars. The adjustment factor is the one in force at
  snapshot time; a snapshot is immutable, so re-running a historical date uses the same
  adjustment.
* A fundamental observation with report period `report_date` becomes usable on its announcement
  date `notice_date`. If the vendor omits `notice_date`, availability is `report_date + 90 days`
  and the row is marked `notice_date_estimated`.
* A monthly signal formed at the close of decision date `T` may use prices with `date <= T` and
  fundamentals available at `T`. Trades execute at the close of the next trading day `T+1`, so
  no return between `T` and `T+1` is attributed to the signal.
* When the announcement date is missing and the estimate is used, the PIT audit records it; the
  verification stage fails the run if a fundamental with `notice_date > T` is found in use.

## 3. Raw factor definitions

All factors are computed cross-sectionally at each month-end. A trailing window of `w` months
uses the last `w` month-end observations including the current one unless stated otherwise.

| Factor id | Definition | Direction |
|---|---|---|
| `value_ep` | trailing-twelve-month EPS / close | + |
| `value_bp` | latest book value per share / close | + |
| `value_spp` | trailing revenue per share / close; revenue per share = total revenue / total shares | + |
| `quality_roe` | TTM net profit / latest book equity, approximated by EPS / BPS | + |
| `quality_gross_margin` | gross profit / revenue from the latest report | + |
| `quality_cashflow` | operating cash flow per share / EPS (cash conversion) | + |
| `growth_revenue` | revenue year-over-year growth from the latest report | + |
| `growth_profit` | net profit year-over-year growth from the latest report | + |
| `momentum_12_1` | price return from month `t-12` to `t-1` (skip the most recent month) | + |
| `reversal_1m` | negative of the most recent one-month return | + (on the negative return) |
| `low_vol_60d` | negative of the 60-trading-day realised volatility of daily returns | + (on low vol) |
| `turnover_20d` | negative of the 20-trading-day average daily turnover | + (on low turnover) |
| `lottery_max_20d` | negative of the maximum daily return in the last 20 trading days | + (on low lottery) |

For value and quality factors the "price" denominator is the decision-date close, so a cheap
stock has a high factor value. Negative stockholder equity or a negative EPS makes the ratio
undefined; such observations are set to missing, not to an extreme value.

## 4. Preprocessing and neutralisation

1. **Winsorisation.** Each raw factor is clipped at the 1st and 99th cross-sectional
   percentiles. This limits the influence of accounting outliers without deleting them.
2. **Rank standardisation.** Values are converted to normal scores: rank / (N + 1), then
   inverse normal CDF. This makes factors comparable across time and robust to distribution
   shifts.
3. **Neutralisation.** For each month, the standardised factor `z` is regressed on industry
   dummies and log market capitalisation,
   `z_i = a + sum_k b_k * industry_{i,k} + c * log_size_i + e_i`,
   and the residual `e_i` is standardised again. The composite therefore measures within-industry,
   size-adjusted factor exposure, which is what an index-enhancement book can trade without
   taking a large unintended sector or size bet.
4. **Composite.** The composite score is the weighted sum of neutralised factor scores with
   weights supplied by the hypothesis and normalised to one over the available factors. Missing
   factor observations are excluded from the sum and the weights renormalised, so a name is not
   penalised for a missing accounting item.
5. **Eligibility.** A name needs at least the configured minimum number of non-missing factors,
   a valid price, no suspension and an active listing. Non-eligible names receive no weight.

The factor report records coverage, rank IC (Spearman correlation between the factor and the
next-month cross-sectional return), the IC t-statistic, first-order autocorrelation and the
cross-factor correlation matrix. These are diagnostics for the analyst and the verifier; they
never feed back into the portfolio automatically.

## 5. Backtest

* Monthly decision dates are the last trading day of each calendar month in the test window.
* The signal is the composite score at `T`; trades execute at the close of `T+1`.
* A-share lot size is 100 shares. A buy is skipped if the stock is limit-up at execution; a sell
  is skipped if the stock is limit-down or suspended. The resulting small cash residual is held
  in cash and is reflected in NAV.
* Costs: commission 2.5 bp per side, stamp duty 5 bp on sells, slippage 5 bp per side, and a
  square-root market-impact term `impact_coef * sqrt(participation) * notional`, where
  participation is the order notional divided by the stock's 20-day average turnover value.
  All coefficients are configurable in `config/strategy.yaml`.
* The ledger records each fill with shares, price, gross amount, each cost component and cash
  flow. Holdings are marked to market daily. The verifier reconstructs NAV from the ledger and
  the price panel and must match the reported path.

Performance metrics: annualised geometric return, annualised volatility, Sharpe ratio (using
the configured risk-free rate, zero by default), annualised excess return over the CPI 300,
tracking error (annualised standard deviation of monthly excess returns), information ratio
(excess return / tracking error), maximum drawdown, maximum excess drawdown, monthly win rate,
average one-way turnover, total cost in basis points, and calendar-year breakdown.

## 6. Portfolio construction

At each rebalance date the optimiser solves

    maximise   alpha' w  -  (lambda / 2) * (w - w_b)' * Sigma * (w - w_b)
    subject to sum_i w_i = 1,  0 <= w_i <= max_weight,
               |w_i - w_b,i| <= max_active_weight,
               |sum_{i in industry k} (w_i - w_b,i)| <= industry_active_limit,
               (w - w_b)' * Sigma * (w - w_b) <= TE_target^2,

where `w_b` are benchmark weights, `alpha` is the neutralised composite score centred on zero,
and `Sigma` is the Ledoit-Wolf shrunk covariance of daily returns over the 120-day lookback,
annualised. The tracking-error constraint is handled by bisection on `lambda`: the optimiser is
solved for a sequence of penalty values, and the value whose ex-ante tracking error is closest to
the target is kept. The industry-neutral box constraints are enforced exactly by projecting
each industry block onto the simplex-with-bounds intersection, which keeps the industry active
weights at zero to numerical precision.

Turnover control is a deterministic blend: the final target is
`w = (1 - kappa) * w_previous + kappa * w_optimised`, with `kappa` the largest value in [0, 1]
that keeps one-way turnover below the configured limit. The blend itself is reported in the risk
report. If the optimiser fails to converge, the stage falls back to rank-proportional weights
within the industry-neutral feasible set, clips to the caps, and sets `fallback_used: true`;
the verifier fails the run if a fallback is used outside the configured tolerance because a
fallback is a materially different portfolio.

Orders are generated from the difference between the target and the previous approved holdings,
rounded to lots, capped at a participation limit of 10% of 20-day average daily volume, and
reported with side, weight change, notional and participation.

## 7. Robustness and multiple testing

The robustness stage runs the following checks on the same frozen data:

| Check | Purpose |
|---|---|
| Sub-period stability | split 2018–2026 into three regimes; a result that exists only in one regime is flagged |
| Cost sensitivity | re-price turnover at 0x, 1x and 3x the base cost; a strategy that dies at 3x is not robust |
| Leave-one-factor-out | re-run the composite without each factor; a single factor dominating the result is flagged |
| Parameter grid | tracking-error target 3%, 5%, 7% and monthly vs quarterly rebalancing |
| Placebo | replace the composite with seeded random scores, 20 repetitions; the real signal must beat the placebo distribution |
| Survivorship stress | remove the 30 names with the highest full-period returns and re-run; the excess return must not collapse |
| Deflated Sharpe ratio | adjust the Sharpe ratio for the number of trials, skewness and kurtosis (Bailey and Lopez de Prado) |

The deflated Sharpe ratio (DSR) is computed as

    DSR = Z( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + ((kurt - 1) / 4) * SR^2) ),

where `SR0` is the expected maximum Sharpe ratio across the number of independent trials, `T`
is the number of observations, and `Z` is the standard normal CDF. It answers the question "is
this Sharpe ratio impressive after accounting for how many ideas were tried?", which is the
main defence against backtest mining.

## 8. Monitoring

Each month the monitoring run compares the realised return of the approved portfolio with the
benchmark since the last rebalance and reports:

* realised tracking error and tracking difference versus the 5% target;
* ex-ante versus ex-post factor exposures and rank-IC drift;
* data freshness (snapshot max date versus the monitoring date) and coverage;
* breaches and a stop rule: two consecutive months with realised tracking error above twice the
  target, or an excess drawdown above the configured limit, stop new rebalances until a human
  reviews the product.

Monitoring reads only the snapshot and the approved portfolio; it never silently revises the
approved weights.

## 9. Known limitations

1. **Survivorship and reconstitution bias.** The universe is the current constituent list, so
   delisted names are absent and additions are known in advance. The bias-stress test bounds the
   effect but cannot eliminate it.
2. **Vendor approximation.** Free-float weights, industry classification and the fallback
   announcement dates are vendor approximations. Published CSI 300 weights and official
   announcement timestamps would be preferable for production.
3. **Single market and period.** The evidence covers one market over one period; regime shifts,
   short-sale constraints and capacity are not modelled.
4. **Model risk.** A local 9B language model can produce weak or inconsistent prose. Its outputs
   are advisory and schema-checked; all numbers and gates are deterministic. The research note
   must be reviewed by a human before publication, which the approval gate enforces.
5. **No live execution.** The package is a research prototype. It does not place orders, does
   not model market impact beyond the square-root approximation and does not claim production
   performance.
