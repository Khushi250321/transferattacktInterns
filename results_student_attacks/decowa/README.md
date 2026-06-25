# DeCowA Student Contribution

This folder contains a verified student-contributed attack adaptation evaluated on the same subset baseline setup used in this repository.

## Contributor
- **Name:** Om Singh Rawat
- **College:** IIT Delhi

## Attack
- **Implementation name:** `DECOWA`
- **Type:** Deformation-constrained warping based transfer attack

## Reference paper
- **Title:** *Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping*
- **Authors:** Jiayi Lin, Chuanbai Xiao, Chao Ma, Jie Zhang, Qiong Cao, Xiaosen Wang
- **Venue:** AAAI 2024
- **Paper:** https://arxiv.org/abs/2402.03951
- **Code:** https://github.com/Trustworthy-AI-Group/TransferAttack

## Important note
The implementation in this repository is a face-verification adaptation integrated into the shared transfer-attack pipeline. The core DeCowA idea is preserved: each iteration averages gradients over multiple thin-plate-spline warped views, with a deformation-constrained inner update for the control-point noise.

## Verified result on the provided subset
- **Overall breach rate:** `32.50%`
- **Mean impact:** `0.1931`
- **Dodging breach rate:** `43.75%`
- **Impersonation breach rate:** `21.25%`

## Comparison against current baseline
Compared with the current official vanilla baseline in this repo, `DECOWA` outperformed all vanilla attacks on the shared subset. It ranked below the verified `BSR` result and above the verified `BPA_CNN` result under the same evaluation setup.
