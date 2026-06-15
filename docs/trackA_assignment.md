# Track A: Transfer-Based Adversarial Attacks (Ms. Pratishtha)

Transfer attacks are adversarial attacks in which an adversarial example is generated using full access to one face recognition model (the surrogate or attacker model) and then tested on a different unseen model (the victim or black-box model).

## Repository for this track
GitHub repository: https://github.com/AItruste/transferattacktInterns

This repository is a **vanilla-only** setup for interns.

## Current experiment setup
### Open-source attacker (surrogate) models
- Facenet512
- ArcFace
- GhostFaceNet
- VGG-Face

### Victim models
- Facenet512
- ArcFace
- GhostFaceNet
- VGG-Face
- IR152

Self-transfer pairs are excluded in evaluation.

## Baseline attacks already implemented and evaluated
- PGD (Projected Gradient Descent)
- MI-FGSM
- TI-FGSM
- SI-NI-FGSM
- MI-ADMIX-DI-TI

## Important note
Your comparison should be against the existing **vanilla attacks only**.

## Smaller subset for this exercise
Use the provided smaller subset first:
- `docs/subset_input_pairs.csv`

Existing baseline references for this subset:
- `results_baseline/subset_raw_similarities_long.csv`
- `results_baseline/subset_attack_eval_long.csv`
- `results_baseline/subset_attack_summary.csv`
- `results_baseline/subset_attack_summary_by_goal.csv`
- `results_baseline/subset_attacker_victim_summary.csv`
- `results_baseline/baseline_notes.md`

Important baseline usage note:
- Please use the provided CSV files in `results_baseline/` as the official baseline reference for comparison.
- If you rerun the baseline attacks locally, you may observe small differences in the reported values.
- This is expected because adversarial sample generation is not fully deterministic across runs and environments.
- Therefore, your new attack should be compared against the provided baseline CSVs, not against a fresh local rerun unless explicitly instructed.

## Current strongest vanilla baselines on this subset
- SI-NI-FGSM
- MI-FGSM

## Your task
Implement and evaluate a newer transfer attack or transfer-improvement strategy that is **not yet implemented in this repo**.

You should:
1. reproduce the existing vanilla baseline results on the small subset,
2. implement one or more newer attacks from recent literature,
3. adapt the attack to the face verification setting,
4. compare against the existing 5 vanilla baselines,
5. analyze whether the new method improves transferability.

## What to compare
You must compare whether your implementation improves:
- breach success rate,
- embedding-level impact,
- cross-model transferability,
- impersonation and dodging performance.

## Upstream source for new attacks
Check the implementation source here:
- https://github.com/Trustworthy-AI-Group/TransferAttack

## Suggested attacks / papers to explore
Focus mainly on recent attacks from **2023 onward** in core ML / vision conferences.

Possible candidates include:
- FPR (2025)
- ATT (NeurIPS 2024)
- BPFA (NeurIPS 2022)
- DPA (CVPR 2025)
- TransMix (2024)
- Ghost-network / surrogate-diversification style transfer methods
- feature-level augmentation or transfer-enhancement methods from recent 2023-2025 literature

## Deliverables
- clean code integrated into the repo or organized clearly,
- result CSV files,
- at least one comparison table or plot,
- a short technical report summarizing:
  - implementation details,
  - hyperparameters,
  - comparison against baselines,
  - strengths / failures,
  - reproducibility notes.

## Additional expectation
Please do not spend time re-implementing the older attacks that are already present here. The main goal is to add a **new** attack and compare it properly against the current vanilla baselines.
