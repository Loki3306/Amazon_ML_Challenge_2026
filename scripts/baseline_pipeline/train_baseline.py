import sys
import os
import time
import json
import argparse
import collections
import numpy as np
import joblib
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Ensure scripts/akash is in python path
src_dir = os.path.abspath(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import normalization as norm
from features import extract_features_for_pair, FEATURE_NAMES
from model import EntityMatcherModel
from evaluation import evaluate_predictions
from thresholding import apply_threshold_and_deduplication
from blocking import get_blocking_keys


def auto_detect_train_dir():
    candidates = ['dataset/train', 'student_resource/dataset/train']
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description='Train Business Entity Resolution Model with Hard Negative Mining')
    parser.add_argument('--train-dir', default=auto_detect_train_dir(),
                        help='Path to train dataset directory')
    parser.add_argument('--val-ids', default=os.path.join(src_dir, 'val_s1_ids.txt'),
                        help='Path to file containing held-out validation Source 1 IDs')
    parser.add_argument('--n-val', type=int, default=10000,
                        help='Number of validation Source 1 entities to evaluate')
    parser.add_argument('--n-train', type=int, default=60000,
                        help='Number of training Source 1 entities to sample')
    parser.add_argument('--n-distractors', type=int, default=300000,
                        help='Number of realistic background distractors to load from target sources')
    parser.add_argument('--model-type', default='xgboost_gpu', choices=['xgboost_gpu', 'xgboost', 'lightgbm'],
                        help='Model architecture to train (xgboost_gpu uses NVIDIA CUDA GPU)')
    parser.add_argument('--n-estimators', type=int, default=500,
                        help='Number of gradient boosted trees')
    parser.add_argument('--model-out', default='models/final_entity_matcher.joblib',
                        help='Output path for trained model')
    parser.add_argument('--meta-out', default='models/model_metadata.json',
                        help='Output path for model metadata')
    args = parser.parse_args()

    print('=== Training Business Entity Resolution Model with Hard Negatives ===')
    print(f'Train directory : {args.train_dir}')
    print(f'Validation split: {args.val_ids} (using {args.n_val:,} entities)')
    print(f'Training sample : {args.n_train:,} S1 entities')

    gt_file = os.path.join(args.train_dir, 'train_ground_truth.tsv')
    s1_file = os.path.join(args.train_dir, 'train_source1.tsv')
    s2_file = os.path.join(args.train_dir, 'train_source2.tsv')
    s3_file = os.path.join(args.train_dir, 'train_source3.tsv')

    if not os.path.isfile(gt_file):
        print(f'Error: Ground truth file not found at {gt_file}')
        sys.exit(1)

    # 1. Load Validation S1 set
    val_s1_ids = []
    if os.path.isfile(args.val_ids):
        with open(args.val_ids, 'r', encoding='utf-8') as f:
            for line in f:
                sid = line.strip()
                if sid:
                    val_s1_ids.append(sid)
                if len(val_s1_ids) >= args.n_val:
                    break
    val_s1_set = set(val_s1_ids)
    print(f'Loaded {len(val_s1_set):,} validation entities.')

    val_gt = {}
    val_true_targets = set()
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in val_s1_set:
                mids = p[1].split(',') if len(p) > 1 and p[1] else []
                val_gt[p[0]] = set(mids)
                val_true_targets.update(mids)

    # 2. Select Training S1 records (strictly disjoint from validation)
    train_s1_ids = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            sid = p[0]
            if sid not in val_s1_set:
                train_s1_ids.append(sid)
                if len(train_s1_ids) >= args.n_train:
                    break

    train_s1_set = set(train_s1_ids)
    train_gt = {}
    train_true_targets = set()
    with open(gt_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in train_s1_set:
                mids = p[1].split(',') if len(p) > 1 and p[1] else []
                train_gt[p[0]] = set(mids)
                train_true_targets.update(mids)

    print(f'Training pool: {len(train_s1_ids):,} S1 records, {sum(len(v) for v in train_gt.values()):,} true positive links.')

    # 3. Load S1 raw records for train and val
    all_needed_s1 = val_s1_set | train_s1_set
    s1_raw = {}
    with open(s1_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in all_needed_s1:
                s1_raw[p[0]] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')

    s1_preprocessed = {}
    s1_blocking_keys = {}
    for sid, (rname, raddr, rcountry) in s1_raw.items():
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, pnum, _ = norm.normalize_address(raddr)
        s1_preprocessed[sid] = (cn, core_n, ca, nums, pnum, rcountry)
        s1_blocking_keys[sid] = get_blocking_keys(rname, raddr, rcountry)

    # 4. Load Target records (True targets + realistic background distractors)
    all_needed_targets = val_true_targets | train_true_targets
    target_raw = {}
    max_d_per_src = args.n_distractors // 2
    d_s2 = 0
    with open(s2_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            tid = p[0]
            if tid in all_needed_targets or d_s2 < max_d_per_src:
                target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
                if tid not in all_needed_targets:
                    d_s2 += 1

    d_s3 = 0
    with open(s3_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            tid = p[0]
            if tid in all_needed_targets or d_s3 < max_d_per_src:
                target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
                if tid not in all_needed_targets:
                    d_s3 += 1

    print(f'Total target pool: {len(target_raw):,} records loaded (including {d_s2 + d_s3:,} distractors).')

    target_preprocessed = {}
    target_keys = {}
    for tid, (rname, raddr, rcountry) in target_raw.items():
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, pnum, _ = norm.normalize_address(raddr)
        target_preprocessed[tid] = (cn, core_n, ca, nums, pnum, rcountry)
        target_keys[tid] = get_blocking_keys(rname, raddr, rcountry)

    # Inverted index by country
    print('Building country inverted indices...')
    indices = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid, tkeys in target_keys.items():
        rcountry = target_preprocessed[tid][5]
        for k in tkeys:
            indices[rcountry][k].append(tid)

    for c in indices:
        for k in list(indices[c].keys()):
            limit = 350 if (k[0].startswith('n') or k[0].startswith('core') or k[0].startswith('compact')) else 150
            if len(indices[c][k]) > limit:
                del indices[c][k]

    def get_candidates(sid, top_k=20):
        s_keys = s1_blocking_keys[sid]
        country = s1_preprocessed[sid][5]
        c_index = indices[country]
        counts = collections.Counter()
        for k in s_keys:
            if k in c_index:
                counts.update(c_index[k])
        if counts:
            return counts.most_common(top_k)
        return []

    # 5. Build Training Feature Matrix with Active Hard Negative Mining
    print('Fitting Global TF-IDF Vocabulary on Names...')
    s1_id_to_idx = {sid: i for i, sid in enumerate(s1_preprocessed.keys())}
    target_id_to_idx = {tid: i for i, tid in enumerate(target_preprocessed.keys())}
    s1_names = [s1_preprocessed[sid][0] for sid in s1_id_to_idx.keys()]
    target_names = [target_preprocessed[tid][0] for tid in target_id_to_idx.keys()]
    
    tfidf_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_df=0.8, min_df=2)
    tfidf_vec.fit(s1_names + target_names)
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf_vec, 'models/tfidf_name.pkl')
    
    print('Transforming TF-IDF matrices...')
    s1_tfidf_mat = tfidf_vec.transform(s1_names)
    target_tfidf_mat = tfidf_vec.transform(target_names)
    
    print('Encoding Semantic Vectors (all-MiniLM-L6-v2)...')
    # Use CPU by default, it takes ~2 mins for 600k strings on Kaggle
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(' -> Encoding target names...')
    t_embeddings = embed_model.encode(target_names, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    print(' -> Encoding S1 names...')
    s1_embeddings = embed_model.encode(s1_names, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    print('Building FAISS Semantic Index...')
    d = t_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(t_embeddings)
    
    print('Retrieving Semantic Candidates...')
    top_k_faiss = 15
    faiss_distances, faiss_indices = faiss_index.search(s1_embeddings, top_k_faiss)
    
    target_ids_list = list(target_id_to_idx.keys())
    faiss_cands = collections.defaultdict(list)
    for i, sid in enumerate(s1_id_to_idx.keys()):
        for j in range(top_k_faiss):
            tid = target_ids_list[faiss_indices[i][j]]
            score = faiss_distances[i][j]
            if score >= 0.70:  # Only keep somewhat confident semantic matches
                faiss_cands[sid].append((tid, score))

    print('Extracting features for training pairs (including hard negatives)...')
    t_feat_start = time.time()
    X_train = []
    y_train = []
    train_target_set = set(target_preprocessed.keys())

    for sid in train_s1_ids:
        true_mids = train_gt.get(sid, set()) & train_target_set
        cands = get_candidates(sid, top_k=20)
        cand_mids = {tid: count for tid, count in cands}
        
        # Inject Semantic Candidates
        for tid, score in faiss_cands[sid]:
            if tid not in cand_mids:
                cands.append((tid, 1)) # Add to candidates list with fake count 1
                cand_mids[tid] = 1

        s1_tup = s1_preprocessed[sid][:5]
        sid_idx = s1_id_to_idx[sid]
        s1_vec = s1_tfidf_mat[sid_idx]

        for mid in true_mids:
            if mid in target_preprocessed:
                t_tup = target_preprocessed[mid][:5]
                sh = cand_mids.get(mid, 1)
                t_idx = target_id_to_idx[mid]
                t_vec = target_tfidf_mat[t_idx]
                tfidf_sim = float(s1_vec.multiply(t_vec).sum())
                semantic_sim = float(np.dot(s1_embeddings[sid_idx], t_embeddings[t_idx]))
                feats = extract_features_for_pair(s1_tup, t_tup, mid, sh, tfidf_sim, semantic_sim)
                X_train.append(feats)
                y_train.append(1)

        # Mine hard negatives:
        # 1. Candidates that share keys with S1 but are NOT true matches
        neg_count = 0
        for tid, sh in cands:
            if tid not in true_mids and tid in target_preprocessed:
                t_tup = target_preprocessed[tid][:5]
                t_idx = target_id_to_idx[tid]
                t_vec = target_tfidf_mat[t_idx]
                tfidf_sim = float(s1_vec.multiply(t_vec).sum())
                semantic_sim = float(np.dot(s1_embeddings[sid_idx], t_embeddings[t_idx]))
                feats = extract_features_for_pair(s1_tup, t_tup, tid, sh, tfidf_sim, semantic_sim)
                X_train.append(feats)
                y_train.append(0)
                neg_count += 1
                if neg_count >= max(3, len(true_mids) * 3):
                    break

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    print(f'Training dataset: X_train shape = {X_train.shape} (Positives = {pos_count:,}, Negatives = {neg_count:,}) in {time.time()-t_feat_start:.1f}s.')

    # 6. Fit Production Model (GPU Accelerated if xgboost_gpu)
    print(f'Training production {args.model_type} model on {"NVIDIA GPU (CUDA)" if "gpu" in args.model_type else "CPU"}...')
    t_train_start = time.time()
    final_model = EntityMatcherModel(
        args.model_type,
        n_estimators=800,
        learning_rate=0.035,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.85
    )
    final_model.fit(X_train, y_train)
    print(f'Model trained in {time.time()-t_train_start:.1f}s.')

    # 7. Evaluate on Large Held-out Validation Set
    print('Evaluating on held-out validation set...')
    val_pair_list = []
    retrieved_val_true = 0
    total_val_true = sum(len(v) for v in val_gt.values())

    for sid in val_s1_ids:
        cands = get_candidates(sid, top_k=20)
        cand_ids = [tid for tid, _ in cands]
        
        # Inject Semantic Candidates for validation recall
        cand_mids = set(cand_ids)
        for tid, score in faiss_cands.get(sid, []):
            if tid not in cand_mids:
                cands.append((tid, 1))
                cand_ids.append(tid)
                cand_mids.add(tid)
                
        retrieved_val_true += len(val_gt[sid] & set(cand_ids))

        s1_tup = s1_preprocessed[sid][:5]
        sid_idx = s1_id_to_idx[sid]
        s1_vec = s1_tfidf_mat[sid_idx]
        
        for tid, sh in cands:
            if tid in target_preprocessed:
                t_tup = target_preprocessed[tid][:5]
                t_idx = target_id_to_idx[tid]
                t_vec = target_tfidf_mat[t_idx]
                tfidf_sim = float(s1_vec.multiply(t_vec).sum())
                semantic_sim = float(np.dot(s1_embeddings[sid_idx], t_embeddings[t_idx]))
                feats = extract_features_for_pair(s1_tup, t_tup, tid, sh, tfidf_sim, semantic_sim)
                val_pair_list.append((sid, tid, feats, s1_tup, t_tup))

    X_val = np.array([p[2] for p in val_pair_list], dtype=np.float32)
    val_probas = final_model.predict_proba(X_val)

    scores_dict = collections.defaultdict(list)
    for (sid, tid, feats, s1_tup, t_tup), p in zip(val_pair_list, val_probas):
        prob = float(p)
        scores_dict[sid].append((tid, prob))

    for sid in val_s1_ids:
        if sid not in scores_dict:
            scores_dict[sid] = []

    # Grid search optimal thresholds with bipartite 1-to-1 consistency
    print('Optimizing source-specific thresholds with 1-to-1 deduplication...')
    best_f05 = -1.0
    best_s2 = 0.80
    best_s3 = 0.88
    best_metrics = None

    for t2 in [0.70, 0.75, 0.80, 0.85]:
        for t3 in [0.75, 0.80, 0.85, 0.90]:
            preds = apply_threshold_and_deduplication(scores_dict, t2, t3)
            metrics = evaluate_predictions(val_gt, preds)
            if metrics['macro_f05'] > best_f05:
                best_f05 = metrics['macro_f05']
                best_s2 = t2
                best_s3 = t3
                best_metrics = metrics

    opt_preds = apply_threshold_and_deduplication(scores_dict, best_s2, best_s3)
    final_metrics = evaluate_predictions(val_gt, opt_preds)
    val_cand_recall = retrieved_val_true / total_val_true if total_val_true > 0 else 0.0

    print('\n' + '=' * 60)
    print('FINAL MODEL VALIDATION BENCHMARKS')
    print('=' * 60)
    print(f"Optimal Thresholds: S2 = {best_s2:.2f}, S3 = {best_s3:.2f}")
    print(f"Validation Macro F0.5 : {final_metrics['macro_f05']:.6f}")
    print(f"Validation Precision  : {final_metrics['global_precision']:.6f}")
    print(f"Validation Recall     : {final_metrics['global_recall']:.6f}")
    print(f"Validation Macro F1   : {final_metrics['macro_f1']:.6f}")
    print(f"Candidate Recall      : {val_cand_recall:.6f}")
    print(f"False Positives       : {final_metrics['total_fp']}")
    print(f"False Negatives       : {final_metrics['total_fn']}")
    print(f"Singleton Accuracy    : {final_metrics['singleton_accuracy']:.6f}")
    print('=' * 60 + '\n')

    # 8. Save Model and Metadata
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or '.', exist_ok=True)
    final_model.save(args.model_out)

    meta = {
        'optimal_s2_threshold': float(best_s2),
        'optimal_s3_threshold': float(best_s3),
        'validation_f05': float(final_metrics['macro_f05']),
        'validation_precision': float(final_metrics['global_precision']),
        'validation_recall': float(final_metrics['global_recall']),
        'validation_f1': float(final_metrics['macro_f1']),
        'candidate_recall': float(val_cand_recall),
        'false_positives': int(final_metrics['total_fp']),
        'false_negatives': int(final_metrics['total_fn']),
        'singleton_accuracy': float(final_metrics['singleton_accuracy']),
        'model_type': args.model_type,
        'feature_names': FEATURE_NAMES,
        'train_samples': len(train_s1_ids),
        'train_positives': pos_count,
        'train_negatives': neg_count,
        'val_samples': len(val_s1_ids)
    }
    with open(args.meta_out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Model saved to {args.model_out} and metadata to {args.meta_out}.')


if __name__ == '__main__':
    main()
