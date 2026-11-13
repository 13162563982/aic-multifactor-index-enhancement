# Registry rollback evidence

`rollback_v1.1.0_to_v1.0.0.json` is the appended rollback event: it records the from/to versions, the bundle integrity + smoke evaluation result, the full before/after headline metrics, the restored config/prompts/skills file list, and the immutable registry entry (sequence, hashes).

`registry_history.json` is the complete append-only history after the rollback (v1.0.0 release -> v1.1.0 release -> rollback event) with the digest chain intact.

The rollback restored the v1.0.0 bundle into a temporary directory so the working tree was not disturbed; the restored file list is recorded in the rollback event.
