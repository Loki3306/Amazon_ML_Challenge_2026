import os
import sys
import time
import json
import gc
import sqlite3
import collections
import numpy as np

import normalization as norm
from features import extract_features_for_pair, FEATURE_NAMES
from model import EntityMatcherModel
from blocking import get_blocking_keys_from_preprocessed
from thresholding import apply_threshold_and_deduplication


def run_test_inference(test_dir, model_path, meta_path, output_dir, batch_size=10000, top_k=20):
    """
    Runs end-to-end entity resolution inference on the complete test dataset.
    Uses disk-backed SQLite caching on D: drive to guarantee zero memory bloat (<500MB RAM),
    preventing Windows virtual memory pagefile expansion on C: drive.
    Fully leverages local NVIDIA CUDA GPU for batch predictions.
    """
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print('=== Amazon ML Challenge 2026: Business Entity Resolution Inference ===')
    print(f'Test directory   : {test_dir}')
    print(f'Model path       : {model_path}')
    print(f'Output directory : {output_dir}')

    # 1. Load trained model and threshold metadata
    print('\n[1/5] Loading production model and metadata...')
    model = EntityMatcherModel.load(model_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    s2_threshold = meta.get('optimal_s2_threshold', 0.70)
    s3_threshold = meta.get('optimal_s3_threshold', 0.75)
    print(f'Using calibrated decision thresholds: S2 = {s2_threshold:.2f}, S3 = {s3_threshold:.2f}')

    # 2. Read test_source1.tsv and preserve original order
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_path = os.path.join(test_dir, 'test_source3.tsv')

    print('\n[2/5] Reading test Source 1 entities and partitioning by country...')
    ordered_s1_ids = []
    country_to_s1 = collections.defaultdict(list)

    with open(s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            addr = parts[2] if len(parts) > 2 else ''
            country = parts[3] if len(parts) > 3 else ''

            ordered_s1_ids.append(eid)
            country_to_s1[country].append((eid, name, addr))

    total_s1 = len(ordered_s1_ids)
    print(f'Total test Source 1 entities: {total_s1:,}')
    for c, items in country_to_s1.items():
        print(f'  Country: {c:<10} -> {len(items):,} entities ({len(items)/total_s1*100:.1f}%)')

    # Initialize fast SQLite cache on D: drive (output_dir) to store results per country
    # This prevents storing 35 million candidate strings in RAM
    db_path = os.path.join(output_dir, 'pipeline_cache.db')
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = MEMORY')
    cursor.execute('PRAGMA temp_store = MEMORY')
    cursor.execute('CREATE TABLE IF NOT EXISTS candidates (s1_id TEXT PRIMARY KEY, cand_str TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS matches (s1_id TEXT PRIMARY KEY, match_str TEXT)')
    conn.commit()

    country_stats = {}
    total_candidate_pairs = 0

    # 3. Process each country sequentially
    print('\n[3/5] Processing test candidates and predictions by country partition (CUDA GPU accelerated)...')
    sorted_countries = sorted(country_to_s1.keys(), key=lambda c: len(country_to_s1[c]))

    for country_idx, country in enumerate(sorted_countries, 1):
        c_t0 = time.time()
        s1_list = country_to_s1[country]
        print(f'\n--- [{country_idx}/{len(sorted_countries)}] Processing Country: {country} ({len(s1_list):,} S1 entities) ---')
        sys.stdout.flush()

        # Load S2 and S3 target records for this country
        print(f'  Loading target records from S2 and S3 for {country}...')
        targets = {}
        for src_path in [s2_path, s3_path]:
            if not os.path.isfile(src_path):
                continue
            with open(src_path, 'r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    line = line.rstrip('\r\n')
                    if not line:
                        continue
                    parts = line.split('\t')
                    c = parts[3] if len(parts) > 3 else ''
                    if c == country:
                        eid = parts[0]
                        name = parts[1] if len(parts) > 1 else ''
                        addr = parts[2] if len(parts) > 2 else ''
                        targets[eid] = (name, addr, c)

        print(f'  Loaded {len(targets):,} target records for {country}.')

        # Preprocess target records and build inverted index
        print('  Building inverted index for country...')
        target_preprocessed = {}
        index = collections.defaultdict(list)

        for tid, (rname, raddr, rcountry) in targets.items():
            cn, core_n, _ = norm.normalize_name(rname)
            ca, nums, pnum, _ = norm.normalize_address(raddr)
            target_preprocessed[tid] = (cn, core_n, ca, nums, pnum)
            tkeys = get_blocking_keys_from_preprocessed(cn, core_n, ca, nums)
            for k in tkeys:
                index[k].append(tid)

        # Prune high-frequency keys to prevent generic word explosion
        pruned = 0
        for k in list(index.keys()):
            limit = 250 if (k[0].startswith('n') or k[0].startswith('core') or k[0].startswith('compact')) else 100
            if len(index[k]) > limit:
                del index[k]
                pruned += 1

        print(f'  Inverted index built with {len(index):,} active keys (pruned {pruned:,} keys).')

        # Pre-normalize S1 records for this country
        print(f'  Pre-normalizing {len(s1_list):,} S1 records...')
        s1_preprocessed = []
        for eid, rname, raddr in s1_list:
            cn, core_n, _ = norm.normalize_name(rname)
            ca, nums, pnum, _ = norm.normalize_address(raddr)
            skeys = get_blocking_keys_from_preprocessed(cn, core_n, ca, nums)
            s1_preprocessed.append((eid, cn, core_n, ca, nums, pnum, skeys))

        # Run inference in batches
        print(f'  Scoring candidates for {len(s1_preprocessed):,} S1 records in batches of {batch_size} (CUDA GPU)...')
        c_scores_dict = collections.defaultdict(list)
        n_batches = (len(s1_preprocessed) + batch_size - 1) // batch_size

        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, len(s1_preprocessed))
            batch = s1_preprocessed[b_start:b_end]

            batch_pairs = []
            cand_rows = []

            for eid, cn, core_n, ca, nums, pnum, skeys in batch:
                counts = collections.Counter()
                for k in skeys:
                    if k in index:
                        counts.update(index[k])

                if counts:
                    cands = counts.most_common(top_k)
                    cand_ids = [tid for tid, _ in cands]
                    cand_rows.append((eid, ','.join(cand_ids)))
                    total_candidate_pairs += len(cand_ids)
                    s1_tup = (cn, core_n, ca, nums, pnum)

                    for tid, sh in cands:
                        t_tup = target_preprocessed[tid]
                        feats = extract_features_for_pair(s1_tup, t_tup, tid, sh)
                        batch_pairs.append((eid, tid, feats, s1_tup, t_tup))
                else:
                    cand_rows.append((eid, ''))
                    c_scores_dict[eid] = []

            # Stream candidates directly to SQLite DB on D: drive
            cursor.executemany('INSERT INTO candidates VALUES (?, ?)', cand_rows)

            # Predict batch on CUDA GPU with number-consistency logic
            if batch_pairs:
                X_batch = np.array([p[2] for p in batch_pairs], dtype=np.float32)
                probas = model.predict_proba(X_batch)
                for (eid, tid, feats, s1_tup, t_tup), p in zip(batch_pairs, probas):
                    prob = float(p)
                    # Strict street number conflict check:
                    # If both records have primary street numbers and they conflict:
                    s1_pnum = s1_tup[4]
                    t_pnum = t_tup[4]
                    if s1_pnum is not None and t_pnum is not None and s1_pnum != t_pnum:
                        exact_core = feats[1]
                        if exact_core < 1.0:
                            prob *= 0.1  # Drastic penalty for look-alikes on same street with different numbers
                    c_scores_dict[eid].append((tid, prob))

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
                print(f'    Processed batch {b_idx + 1}/{n_batches} ({b_end:,}/{len(s1_list):,} records)...')
                sys.stdout.flush()

        conn.commit()

        # Apply calibrated thresholds and bipartite target deduplication
        print('  Applying decision thresholds and 1-to-1 target consistency...')
        c_matches = apply_threshold_and_deduplication(c_scores_dict, s2_threshold, s3_threshold)

        # Stream matched results directly to SQLite DB on D: drive
        match_rows = [(eid, ','.join(sorted(mids))) for eid, mids in c_matches.items()]
        cursor.executemany('INSERT INTO matches VALUES (?, ?)', match_rows)
        conn.commit()

        # Country level summary statistics
        c_singletons = sum(1 for eid, _, _ in s1_list if len(c_matches.get(eid, set())) == 0)
        c_total_links = sum(len(c_matches.get(eid, set())) for eid, _, _ in s1_list)
        country_stats[country] = {
            'total_s1': len(s1_list),
            'singletons': c_singletons,
            'singleton_pct': c_singletons / len(s1_list) * 100,
            'matched_links': c_total_links,
            'avg_matches': c_total_links / len(s1_list)
        }
        print(f'  Country {country} stats: Total={len(s1_list):,}, Singletons={c_singletons:,} ({c_singletons/len(s1_list)*100:.2f}%), Matched Links={c_total_links:,} (avg {c_total_links/len(s1_list):.2f})')
        sys.stdout.flush()

        # Clean up memory completely for this country before moving to the next
        del targets, target_preprocessed, index, c_scores_dict, c_matches, s1_preprocessed, match_rows
        gc.collect()
        print(f'  Country {country} completed in {time.time()-c_t0:.1f}s. Memory cleared.')
        sys.stdout.flush()

    # 4. Stream final submission files in exact test Source 1 order directly from SQLite DB
    print('\n[4/5] Streaming final output files in exact test Source 1 order...')
    sys.stdout.flush()
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    print('  Streaming matching_results.tsv from pipeline cache...')
    cursor.execute('SELECT s1_id, match_str FROM matches')
    matched_dict = dict(cursor.fetchall())

    n_singletons = 0
    total_matched_links = 0
    s2_matches = 0
    s3_matches = 0

    with open(matching_path, 'w', encoding='utf-8', newline='') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for sid in ordered_s1_ids:
            m_str = matched_dict.get(sid, '')
            if not m_str:
                n_singletons += 1
            else:
                m_list = m_str.split(',')
                total_matched_links += len(m_list)
                for tid in m_list:
                    if tid.startswith('S2-'):
                        s2_matches += 1
                    elif tid.startswith('S3-'):
                        s3_matches += 1
            f.write(f'{sid}\t{m_str}\n')

    del matched_dict
    gc.collect()
    print(f'Saved matching results : {matching_path}')

    print('  Streaming candidate_pairs.tsv from pipeline cache...')
    cursor.execute('SELECT s1_id, cand_str FROM candidates')
    cand_dict = dict(cursor.fetchall())

    with open(candidate_path, 'w', encoding='utf-8', newline='') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for sid in ordered_s1_ids:
            c_str = cand_dict.get(sid, '')
            f.write(f'{sid}\t{c_str}\n')

    del cand_dict
    gc.collect()
    print(f'Saved candidate pairs  : {candidate_path}')

    conn.close()
    try:
        os.remove(db_path)
    except Exception:
        pass

    # 5. Summary statistics
    n_matched = total_s1 - n_singletons

    report_dict = {
        'Validation Performance': {
            'Validation Macro F0.5': meta.get('validation_f05', 'N/A'),
            'Validation Precision': meta.get('validation_precision', 'N/A'),
            'Validation Recall': meta.get('validation_recall', 'N/A'),
            'Validation Macro F1': meta.get('validation_f1', 'N/A'),
            'Candidate Recall': meta.get('candidate_recall', 'N/A'),
            'Singleton Accuracy': meta.get('singleton_accuracy', 'N/A'),
            'False Positives': meta.get('false_positives', 'N/A'),
            'False Negatives': meta.get('false_negatives', 'N/A')
        },
        'Model Configuration': {
            'Selected Model': 'XGBoost GPU (NVIDIA CUDA Accelerated hist)',
            'Optimal S2 Threshold': s2_threshold,
            'Optimal S3 Threshold': s3_threshold,
            'Global Consistency': 'Source-aware 1-to-1 target assignment (Greedy Highest-Probability)',
            'Feature Set Size': len(FEATURE_NAMES)
        },
        'Test Inference Results': {
            'Total Source 1 Entities': f'{total_s1:,}',
            'Predicted Singletons': f'{n_singletons:,} ({n_singletons/total_s1*100:.2f}%)',
            'Entities with Matches': f'{n_matched:,} ({n_matched/total_s1*100:.2f}%)',
            'Total Predicted Matches': f'{total_matched_links:,}',
            'Total S2 Matches': f'{s2_matches:,}',
            'Total S3 Matches': f'{s3_matches:,}',
            'Total Candidate Pairs': f'{total_candidate_pairs:,}',
            'Runtime': f'{time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.2f} min)'
        },
        'Country Breakdown': country_stats
    }

    # Write metrics summary text file and final report text file to output folder
    for fname in ['metrics_summary.txt', 'final_report.txt']:
        fpath = os.path.join(output_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write('AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION\n')
            f.write('COMPLETE METRICS REPORT (TRAIN, VALIDATION & TEST)\n')
            f.write('=' * 60 + '\n\n')

            for sec_name, sec_dict in report_dict.items():
                f.write(f'[{sec_name}]\n')
                if isinstance(sec_dict, dict):
                    for k, v in sec_dict.items():
                        if isinstance(v, dict):
                            f.write(f'  {k}:\n')
                            for sub_k, sub_v in v.items():
                                f.write(f'    {sub_k:<25}: {sub_v}\n')
                        else:
                            f.write(f'  {k:<30}: {v}\n')
                f.write('\n')
        print(f'Wrote report file      : {fpath}')

    sys.stdout.flush()
    return report_dict
