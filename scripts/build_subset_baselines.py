#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ATTACKERS = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face']
VICTIMS = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face', 'IR152']
ATTACKS = ['PGD', 'MI_FGSM', 'TI_FGSM', 'SI_NI_FGSM', 'MI_ADMIX_DI_TI', 'PGN', 'VMI_FGSM']


def equivalent_models(a, b):
    na = str(a).strip().lower().replace('-', '').replace('_', '')
    nb = str(b).strip().lower().replace('-', '').replace('_', '')
    return na == nb


def success(sim, threshold, attack_type):
    return int(sim >= threshold) if str(attack_type).strip().lower() == 'impersonation_attack' else int(sim < threshold)


def impact(clean, adv, attack_type):
    return (adv - clean) if str(attack_type).strip().lower() == 'impersonation_attack' else (clean - adv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-long-csv', required=True)
    ap.add_argument('--input-csv', required=True)
    ap.add_argument('--thresholds-json', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--pairs-per-dataset-per-goal', type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = pd.read_csv(args.input_csv)
    subset_parts = []
    for attack_type in ['impersonation_attack', 'dodging_attack']:
        for dataset in ['lfw_pairs', 'celeba_pairs', 'vggface2_pairs']:
            subset_parts.append(inp[(inp['attack_type'] == attack_type) & (inp['dataset'] == dataset)].head(args.pairs_per_dataset_per_goal))
    subset = pd.concat(subset_parts, ignore_index=True)[['row_id', 'img1', 'img2', 'dataset', 'attack_type']]
    subset.to_csv(out_dir / 'subset_input_pairs.csv', index=False)

    with open(args.thresholds_json) as f:
        thresholds = json.load(f)
    filtered_thresholds = {k: thresholds[k] for k in VICTIMS if k in thresholds}

    long_df = pd.read_csv(args.raw_long_csv)
    rows = list(subset['row_id'])
    keep = long_df[
        long_df['row_id'].isin(rows)
        & long_df['attacker_model'].isin(ATTACKERS)
        & long_df['victim_model'].isin(VICTIMS)
        & long_df['attack_method'].isin(ATTACKS + ['clean'])
        & long_df['variant'].isin(['vanilla', 'clean'])
    ].copy()
    keep = keep[~keep.apply(lambda r: equivalent_models(r['attacker_model'], r['victim_model']) and r['attack_method'] != 'clean', axis=1)]
    keep.to_csv(out_dir / 'subset_raw_similarities_long.csv', index=False)

    clean = keep[keep['attack_method'] == 'clean'][['row_id','attacker_model','victim_model','dataset','attack_type','similarity']].rename(columns={'similarity':'clean_similarity'})
    adv = keep[(keep['variant'] == 'vanilla') & (keep['attack_method'] != 'clean')].copy()
    merged = adv.merge(clean, on=['row_id','attacker_model','victim_model','dataset','attack_type'], how='left')
    merged['threshold'] = merged.apply(lambda r: filtered_thresholds[r['victim_model']][r['dataset']]['threshold'], axis=1)
    merged['breach'] = merged.apply(lambda r: success(r['similarity'], r['threshold'], r['attack_type']), axis=1)
    merged['impact'] = merged.apply(lambda r: impact(r['clean_similarity'], r['similarity'], r['attack_type']), axis=1)
    merged = merged.rename(columns={'similarity':'adv_similarity'})
    merged.to_csv(out_dir / 'subset_attack_eval_long.csv', index=False)

    merged.groupby('attack_method').agg(num_rows=('breach','size'), breach_rate_pct=('breach', lambda s: 100.0 * s.mean()), impact_mean=('impact','mean')).reset_index().sort_values('breach_rate_pct', ascending=False).to_csv(out_dir / 'subset_attack_summary.csv', index=False)
    merged.groupby(['attack_type','attack_method']).agg(num_rows=('breach','size'), breach_rate_pct=('breach', lambda s: 100.0 * s.mean()), impact_mean=('impact','mean')).reset_index().sort_values(['attack_type','breach_rate_pct'], ascending=[True,False]).to_csv(out_dir / 'subset_attack_summary_by_goal.csv', index=False)
    merged.groupby(['attacker_model','victim_model','attack_method']).agg(num_rows=('breach','size'), breach_rate_pct=('breach', lambda s: 100.0 * s.mean()), impact_mean=('impact','mean')).reset_index().sort_values(['attacker_model','victim_model','breach_rate_pct'], ascending=[True,True,False]).to_csv(out_dir / 'subset_attacker_victim_summary.csv', index=False)


if __name__ == '__main__':
    main()
