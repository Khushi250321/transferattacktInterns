import os, sys, json
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, 'core')
from transfer_attack_core import *

DATASET_ROOT = r'C:\transferattacktInterns\dataset_extractedfaces'
SUBSET_CSV = r'docs\subset_input_pairs.csv'
THRESHOLD_FILE = r'thresholds.json'
OUTPUT_CSV = r'results_att\att_results.csv'

os.makedirs('results_att', exist_ok=True)
pairs = pd.read_csv(SUBSET_CSV)
with open(THRESHOLD_FILE) as f:
    thresholds = json.load(f)

results = []
total = len(pairs)
print(f"Running ATT on {total} pairs...")

for i, row in pairs.iterrows():
    print(f"Pair {i+1}/{total}", end='\r')
    src_path = resolve_image_path(row['img1'], DATASET_ROOT)
    tgt_path = resolve_image_path(row['img2'], DATASET_ROOT)
    attack_type = row['attack_type']

    for attacker in ATTACKER_MODELS:
        victim_list = [v for v in VICTIM_MODELS if not (
            (attacker == 'Facenet512' and v in ['Facenet512','Facenet']) or
            (attacker != 'Facenet512' and v == attacker)
        )]
        try:
            input_size = ATTACKER_MODELS[attacker]
            model = build_attacker(attacker)
            src = tf.expand_dims(
                tf.constant(load_and_preprocess(src_path, input_size)), 0)
            tgt = tf.expand_dims(
                tf.constant(load_and_preprocess(tgt_path, input_size)), 0)
            tgt_emb = compute_embedding(model, tgt)
            adv = att_attack(model, src, tgt_emb, attack_type)
            adv_img = denormalize(adv.numpy()[0])

            for victim in victim_list:
                try:
                    v_size = ATTACKER_MODELS.get(victim, (160,160))
                    v_model = build_attacker(victim)
                    adv_resized = tf.expand_dims(
                        tf.constant(load_and_preprocess(
                            src_path, v_size)), 0)
                    clean_emb = compute_embedding(v_model, adv_resized)
                    adv_tf = tf.image.resize(
                        tf.cast(adv_img, tf.float32)[None]/127.5 - 1.0,
                        v_size)
                    adv_emb = compute_embedding(v_model, adv_tf)
                    tgt_v = tf.expand_dims(
                        tf.constant(load_and_preprocess(
                            tgt_path, v_size)), 0)
                    tgt_emb_v = compute_embedding(v_model, tgt_v)
                    clean_sim = float(tf.reduce_sum(
                        clean_emb * tgt_emb_v))
                    adv_sim = float(tf.reduce_sum(
                        adv_emb * tgt_emb_v))
                    thr = thresholds.get(victim,{}).get(
                        'DigiFace',{}).get('threshold', 0.5)
                    if attack_type == 'impersonation_attack':
                        breach = adv_sim >= thr
                    else:
                        breach = adv_sim < thr
                    results.append({
                        'row_id': row.get('row_id', i),
                        'attacker': attacker,
                        'victim': victim,
                        'attack_type': attack_type,
                        'clean_sim': round(clean_sim, 4),
                        'adv_sim': round(adv_sim, 4),
                        'threshold': round(thr, 4),
                        'breach': breach,
                        'impact': round(adv_sim - clean_sim, 4)
                    })
                except Exception as e:
                    print(f"\nVictim error {victim}: {e}")
        except Exception as e:
            print(f"\nAttacker error {attacker}: {e}")

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Results saved to {OUTPUT_CSV}")
print(df.groupby('attack_type')['breach'].mean().mul(100).round(2))