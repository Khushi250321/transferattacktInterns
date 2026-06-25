# LI_BOOST_MI Student Contribution

This folder contains a verified student-contributed attack adaptation evaluated on the same subset baseline setup used in this repository.

## Contributor
- **Name:** Charushi
- **College:** IGDTUW

## Attack
- **Implementation name:** `LI_BOOST_MI`
- **Type:** MI-style logarithmic-shift boosting transfer attack

## Reference basis
- **Student submission basis:** logarithmic-shift boosting MI variant adapted to the face-verification pipeline
- **Note:** This repository records the verified implementation and results used in the shared subset evaluation

## Important note
The verified integration in this repository includes only the `LI_BOOST_MI` variant from the student submission. The BSR-based hybrid variants from the student branch are intentionally not included in the shared main pipeline.

## Verified result on the provided subset
- **Overall breach rate:** `35.21%`
- **Mean impact:** `0.2007`
- **Dodging breach rate:** `46.67%`
- **Impersonation breach rate:** `23.75%`

## Comparison against current baseline
Compared with the current official vanilla baseline in this repo, `LI_BOOST_MI` outperformed all vanilla attacks on the same subset and currently ranks among the strongest verified student-contributed attacks in the shared pipeline.
