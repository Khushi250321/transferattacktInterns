# ATT_CNN Student Contribution

This folder contains a verified student-contributed CNN-side attack adaptation evaluated on the same subset baseline setup used in this repository.

## Contributor
- **Name:** Keshav Raj
- **College:** IIIT Delhi

## Attack
- **Implementation name:** `ATT_CNN`
- **Type:** ATT-inspired CNN gradient-shaping transfer attack

## Reference note
- **Inspiration source:** *ATT* (NeurIPS 2024)

## Important note
This implementation should be treated as a CNN-side adaptation inspired by ATT, not as an official reproduction of the original ATT method. The official ATT paper is built around Vision Transformer token-level operations and backward hooks on attention, QKV, and MLP blocks. The implementation here instead applies gradient-variance normalization, strong-gradient reweighting, and momentum updates directly in the CNN face-verification pipeline.

## Verified result on the provided subset
- **Overall breach rate:** `26.67%`
- **Mean impact:** `0.1646`
- **Dodging breach rate:** `35.00%`
- **Impersonation breach rate:** `18.33%`

## Comparison against current baseline
Compared with the current official vanilla baseline in this repo, `ATT_CNN` is tied with `MI_FGSM` on breach rate and remains below `SI_NI_FGSM`.
