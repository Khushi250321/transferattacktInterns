# BSR Student Contribution

This folder contains a verified student-contributed attack adaptation evaluated on the same subset baseline setup used in this repository.

## Contributor
- **Name:** Chirag Sharma
- **College:** IIIT Vadodara

## Attack
- **Implementation name:** `BSR`
- **Type:** Block shuffle and rotation based transfer attack

## Reference paper
- **Title:** *Boosting Adversarial Transferability by Block Shuffle and Rotation*
- **Authors:** Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang
- **Venue:** CVPR 2024
- **Paper:** https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Boosting_Adversarial_Transferability_by_Block_Shuffle_and_Rotation_CVPR_2024_paper.pdf
- **Code:** https://github.com/Trustworthy-AI-Group/BSR

## Important note
The implementation in this repository is a face-verification adaptation integrated into the shared transfer-attack pipeline. The core BSR idea is preserved: each iteration averages gradients over multiple shuffled-and-rotated block transformations of the current adversarial sample.

## Verified result on the provided subset
- **Overall breach rate:** `36.46%`
- **Mean impact:** `0.2048`
- **Dodging breach rate:** `49.17%`
- **Impersonation breach rate:** `23.75%`

## Comparison against current baseline
Compared with the current official vanilla baseline in this repo, `BSR` outperformed all existing vanilla attacks and also ranked above the previously integrated `BPA_CNN` result on the same subset.
