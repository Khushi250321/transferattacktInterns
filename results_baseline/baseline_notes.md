# Baseline Notes

Subset size: 30 source-target pairs

Current strongest vanilla attack on this subset: **SI_NI_FGSM** with breach rate **29.17%** and mean impact **0.1755**.

Important note:
- The CSV files in this folder are the official baseline reference for the intern assignment.
- These results come from a precomputed raw-similarity source and should be used as the comparison baseline.
- If the attacks are regenerated locally, small differences in final breach rate and impact may appear.
- This happens because the adversarial generation step is not fully deterministic across runs and environments.
- Clean similarities should remain effectively unchanged; the observed variation comes mainly from the generated adversarial examples.
