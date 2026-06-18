# SIA_MI_TI Student Contribution

This folder contains a verified student-contributed attack adaptation evaluated on the same subset baseline setup used in this repository.

## Contributor
- **Name:** Janhavi Kishor

## Attack
- **Implementation name:** `SIA_MI_TI`
- **Type:** Structure-invariant transformation attack combined with momentum and translation invariance

## Reference paper
- **Title:** *Improving the Transferability of Adversarial Examples via Structure Invariant Transformation*
- **Venue:** ICCV 2023
- **Paper:** https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Improving_the_Transferability_of_Adversarial_Examples_via_Structure_Invariant_Transformation_ICCV_2023_paper.pdf

## Important note
The implementation in this repository is a face-verification adaptation integrated into the shared transfer-attack pipeline. It combines the structure-invariant block transformation idea with MI-FGSM style momentum and TI-FGSM style gradient smoothing.

## Verified result on the provided subset
- **Overall breach rate:** `23.33%`
- **Mean impact:** `0.1376`
- **Dodging breach rate:** `28.75%`
- **Impersonation breach rate:** `17.92%`

## Comparison against current baseline
Compared with the current official vanilla baseline in this repo, `SIA_MI_TI` ranked below `MI_ADMIX_DI_TI` and above `TI_FGSM` on the shared subset.
