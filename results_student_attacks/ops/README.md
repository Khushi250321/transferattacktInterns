# OPS Student Contribution

This folder records a student-contributed attack adaptation that has been added to the shared transfer-attack pipeline, but has not yet been fully verified on the provided subset.

## Contributor
- **Name:** Kkartik Aggarwal
- **College:** Delhi Technological University (DTU)

## Attack
- **Implementation name:** `OPS`
- **Type:** Operator-perturbation-based stochastic optimization transfer attack

## Reference paper
- **Title:** *Boosting Adversarial Transferability through Augmentation in Hypothesis Space*
- **Venue:** CVPR 2025

## Important note
The implementation has been integrated into `core/transfer_attack_core.py` as a face-verification adaptation of the submitted OPS attack. At the time of this update, the attack has **not** been fully verified on the shared subset because the submitted configuration is computationally very expensive and could not be completed within the available verification time.

## Verification status
- **Status:** Not yet verified
- **Reason:** Full subset verification was not completed due to time and runtime constraints on the submitted OPS configuration.

## Usage note
This entry is kept for code availability, attribution, and future verification. Reported numbers from the student submission should not be treated as official repository results until a full rerun is completed.
