# BPA-CNN Student Contribution

This folder contains a verified student-contributed attack adaptation evaluated on the same subset baseline setup used in this repository.

## Contributor
- **Name:** Om Singh Rawat
- **College:** IIT Delhi

## Attack
- **Implementation name:** `BPA_CNN`
- **Type:** BPA-inspired CNN adaptation for face-verification transfer attacks

## Reference paper
- **Title:** *Rethinking the Backward Propagation for Adversarial Transferability*
- **Authors:** Xiaosen Wang, Kangheng Tong, Kun He
- **Venue:** NeurIPS 2023
- **Paper:** https://proceedings.neurips.cc/paper_files/paper/2023/file/05fe0c633ae41756540dba2a99a36306-Paper-Conference.pdf
- **Code:** https://github.com/Trustworthy-AI-Group/BPA

## Important note
The implementation in this repository is a **BPA-inspired adaptation** for pretrained CNN face-recognition models. It is not a layer-level reimplementation of the original BPA method. Instead, it applies smoothing ideas at the input-gradient level to fit the current face-verification transfer framework.

## Verified result on the provided subset
- **Overall breach rate:** `30.21%`
- **Mean impact:** `0.1803`
- **Dodging breach rate:** `40.00%`
- **Impersonation breach rate:** `20.42%`

## Comparison against current baseline
Compared with the current official vanilla baseline in this repo, `BPA_CNN` outperformed the strongest baseline overall (`SI_NI_FGSM` at `29.17%`) and ranked first on combined breach rate.
