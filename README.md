# AIC final project — multi-factor CSI 300 index-enhancement agent workflow

**Author:** Zhengjia Zhu (student ID 25210660121)
**Course:** AI Coding in Finance, Fall 2026 — Final Project
**Repository:** https://github.com/13162563982/aic-multifactor-index-enhancement

> Status: implementation in progress. This README is replaced with the complete
> file guide, setup instructions and walkthrough before submission.

This project designs and implements a long-running agent workflow for a substantial
recurring buy-side task: **monthly multi-factor index-enhancement research on the
CSI 300**. The workflow connects hypothesis generation, real public market data,
factor construction, backtesting, robustness checks, constrained portfolio
construction, independent verification, human approval, version registry and
monthly monitoring. It keeps persistent state in SQLite, resumes interrupted runs
without repeating completed work, prevents future-information leakage, and tests
proposed changes against a fixed baseline with recorded accept/reject/rollback.

Deliverables include the design note and annotated architecture diagram in `docs/`,
role prompts in `prompts/`, reusable skills in `skills/`, all configuration in
`config/`, the runnable workflow in `aicwf/`, a frozen real-data snapshot and
shareable example outputs, and the verification/recovery evidence in `evidence/`.

_A detailed file guide and run instructions will be filled in once the implementation
is verified end to end._
