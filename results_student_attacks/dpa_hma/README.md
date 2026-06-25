# DPA_HMA Student Contribution

This folder contains a verified student-contributed attack adaptation evaluated on the same subset baseline setup used in this repository.

## Contributor
- **Name:** Kushal Khemka
- **College:** Delhi Technological University (DTU)

## Attack
- **Implementation name:** `DPA_HMA`
- **Type:** Diverse-parameters augmentation with hard-view momentum transfer attack

## Reference paper
- **Title:** *Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation*
- **Venue:** CVPR 2025

## Important note
The implementation in this repository is a face-verification adaptation integrated into the shared transfer-attack pipeline. The core idea is to build diverse affine-transformed hard views of the current adversarial sample, average their gradient signal, and update the perturbation with momentum.

## Verified result on the provided subset
- **Overall breach rate:** `40.21%`
- **Mean impact:** `0.2150`
- **Dodging breach rate:** `51.25%`
- **Impersonation breach rate:** `29.17%`

## Comparison against current baseline
Compared with the current official vanilla baseline in this repo, `DPA_HMA` outperformed all existing vanilla attacks and also ranked above all previously verified student-contributed attacks on the same subset.
