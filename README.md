# transferattacktInterns

A clean intern-facing repository for a **vanilla transfer-attack** exercise on face verification.

This repo is intentionally a standalone vanilla transfer-attack exercise.

## What is included
- a vanilla-only attack core with 5 implemented baseline attacks
- a small curated subset for quick experiments
- existing baseline results on that subset
- a Track A assignment note for implementing a **new** transfer attack from recent literature

## Implemented baseline attacks
- PGD
- MI-FGSM
- TI-FGSM
- SI-NI-FGSM
- MI-ADMIX-DI-TI

## Verified student-contributed attack
- BPA-CNN (Om Singh Rawat, IIT Delhi)
- BSR (Chirag Sharma, IIIT Vadodara)

## Current official subset baseline
- SI_NI_FGSM: 29.17%
- MI_FGSM: 26.67%
- MI_ADMIX_DI_TI: 24.17%
- TI_FGSM: 20.42%
- PGD: 16.67%

## Current verified student results
- BSR (Chirag Sharma, IIIT Vadodara): 36.46% breach rate, 0.2048 mean impact
- This currently ranks first among the verified student-contributed attacks on the provided subset.
- BPA_CNN (Om Singh Rawat, IIT Delhi): 30.21% breach rate, 0.1803 mean impact
- This also ranks above the strongest vanilla baseline on the provided subset.

## Not included
- additional objective-level modifications from other project branches
- API-specific evaluation code paths

## Current experiment setup
### Attacker (surrogate) models
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

Victim evaluation excludes self-transfer pairs.

## Small working subset
See:
- `docs/subset_input_pairs.csv`
- `results_baseline/subset_raw_similarities_long.csv`
- `results_baseline/subset_attack_summary.csv`
- `results_baseline/subset_attack_summary_by_goal.csv`
- `results_baseline/subset_attacker_victim_summary.csv`

## Baseline reproducibility note
- The baseline CSV files included in `results_baseline/` are the official reference for this repo.
- These baseline summaries were prepared from a precomputed raw-similarity source and are the values students should use for comparison.
- If you rerun the attack generation pipeline locally, you may observe small differences in breach rate and impact because adversarial sample generation is not fully deterministic across runs and environments.
- In particular, clean similarities are expected to stay essentially unchanged, while adversarial similarities may vary slightly.

## Goal for interns
Implement one new transfer attack that is **not already present in this repo**, adapt it to the face-verification setting, and compare it against the 5 vanilla baselines using breach rate and impact on the provided subset.

## Upstream attack source to inspect
- https://github.com/Trustworthy-AI-Group/TransferAttack

## Recommended starting point
Read:
- `docs/trackA_assignment.md`
- `core/README.md`
- `results_baseline/baseline_notes.md`
- `results_student_attacks/bsr/README.md`
- `results_student_attacks/bsr/bsr_vs_current_baseline_summary.csv`
- `results_student_attacks/bpa_cnn/README.md`
- `results_student_attacks/bpa_cnn/bpa_cnn_vs_current_baseline_summary.csv`
