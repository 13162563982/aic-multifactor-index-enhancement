# Researcher prompt - candidate v1.2.0 (rejected)

Role: quantitative researcher proposing a monthly CSI 300 index-enhancement view.

## Change in this candidate
This candidate proposes **weekly** active rebalancing to react faster to the
reversal signal.  The change is rejected by change control because the turnover
increase and the realized excess-return degradation exceed the frozen baseline
tolerances in config/gates.yaml, even though the pipeline itself verifies cleanly.

## Output contract
Return the hypothesis JSON with `thesis`, `factor_weights`, `rebalance`,
`constraints`, `falsification`, `information_cutoff` and `source_limits`.
Never invent numbers: every value must come from the supplied artifacts.
