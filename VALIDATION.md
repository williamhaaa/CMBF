# Verification

Verified on 2026-11-24 with Python 3.10.18.

- Complete original training import tree loaded successfully.
- CPU synthetic smoke test exercised linear, node/hypergraph and semantic event branches with finite forward outputs and auxiliary losses.
- Backward produced finite, nonzero gradients in all three branches.
- Evaluation output shape was `[2, 12, 4]`.
- The unchanged `run_STHyper1.py` completed one full training epoch, validation, checkpoint selection and test evaluation on a temporary 256-row synthetic four-variable CSV. These synthetic scores are only an execution check, not a benchmark result.
- The documented Weather command also completed a full training/validation/test cycle on a synthetic 1,000-row, 21-variable, 10-minute CSV, with only data/output paths, CPU selection and the epoch count overridden.
- Every original source file was compared byte-for-byte with the source snapshot and verified against `SOURCE_PROVENANCE.json`.
- No datasets, weights, experiment logs or private credentials are part of the release package.

Verified dependency versions:

```json
{
  "torch": "2.7.0+cu128",
  "torch-geometric": "2.6.1",
  "numpy": "2.1.2",
  "pandas": "2.3.1",
  "scikit-learn": "1.6.1",
  "matplotlib": "3.10.5",
  "tensorboard": "2.20.0"
}
```
