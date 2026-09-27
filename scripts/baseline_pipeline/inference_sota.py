import os
import argparse
import time
import math
import collections
import json
import joblib
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from joblib import Parallel, delayed

import normalization as norm
from features import extract_features_for_pair
from model import EntityMatcherModel
from blocking import get_blocking_keys
from thresholding import apply_threshold_and_deduplication
from output import write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--output-dir', default='output')
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    print('=== Amazon ML Challenge 2026: SOTA Inference ===')
    s1_file = os.path.join(args.test_dir, 'test_source1.tsv')
    s2_file = os.path.join(args.test_dir, 'test_source2.tsv')
    s3_file = os.path.join(args.test_dir, 'test_source3.tsv')

    print('\n[1/5] Loading production model and metadata...')
    model_path = os.path.join(args.model_dir, 'final_entity_matcher.joblib')
    meta_path = os.path.join(args.model_dir, 'model_metadata.json')
    tfidf_path = os.path.join(args.model_dir, 'tfidf_name.pkl')

    final_model = EntityMatcherModel.load(model_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    tfidf_vec = joblib.load(tfidf_path)

    s2_threshold = meta.get('optimal_s2_threshold', 0.40)
    s3_threshold = meta.get('optimal_s3_threshold', 0.40)
    print(f'Using calibrated decision thresholds: S2 = {s2_threshold:.2f}, S3 = {s3_threshold:.2f}')

    print('\n[2/5] Loading Test Target Entities (S2 & S3)...')
    target_raw = {}
    for s_file in [s2_file, s3_file]:
        with open(s_file, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.rstrip('\r\n').split('\t')
                tid = p[0]
                target_raw[tid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')

    print(f'Loaded {len(target_raw):,} target records.')
    target_preprocessed = {}
    target_keys = {}
    target_combined = []
    target_addrs = []
    target_ids_list = []
    
    for tid, (rname, raddr, rcountry) in target_raw.items():
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, pnum, _ = norm.normalize_address(raddr)
        target_preprocessed[tid] = (cn, core_n, ca, nums, pnum, rcountry)
        target_keys[tid] = get_blocking_keys(rname, raddr, rcountry)
        target_combined.append(f"{cn} {ca} {rcountry}")
        target_addrs.append(ca)
        target_ids_list.append(tid)

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

    def get_candidates(s_keys, country, top_k=250):
        c_index = indices[country]
        counts = collections.Counter()
        for k in s_keys:
            if k in c_index:
                counts.update(c_index[k])
            if counts:
                return counts.most_common(top_k)
        return []

    print('\n[3/5] Loading Test Source 1 Entities...')
    s1_raw = {}
    test_s1_ids = []
    with open(s1_file, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            sid = p[0]
            s1_raw[sid] = (p[1] if len(p) > 1 else '', p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else '')
            test_s1_ids.append(sid)

    s1_preprocessed = {}
    s1_blocking_keys = {}
    s1_combined = []
    s1_addrs = []
    for sid, (rname, raddr, rcountry) in s1_raw.items():
        cn, core_n, _ = norm.normalize_name(rname)
        ca, nums, pnum, _ = norm.normalize_address(raddr)
        s1_preprocessed[sid] = (cn, core_n, ca, nums, pnum, rcountry)
        s1_blocking_keys[sid] = get_blocking_keys(rname, raddr, rcountry)
        s1_combined.append(f"{cn} {ca} {rcountry}")
        s1_addrs.append(ca)

    print('Encoding Semantic Vectors (all-MiniLM-L6-v2) on GPU...')
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    embed_model.half()
    
    t_emb_path = os.path.join(cache_dir, 'test_t_embeddings.npy')
    t_addr_emb_path = os.path.join(cache_dir, 'test_t_addr_embeddings.npy')
    s1_emb_path = os.path.join(cache_dir, 'test_s1_embeddings.npy')
    s1_addr_emb_path = os.path.join(cache_dir, 'test_s1_addr_embeddings.npy')
    
    if os.path.exists(t_emb_path):
        t_embeddings = np.load(t_emb_path)
        t_addr_embeddings = np.load(t_addr_emb_path)
    else:
        t_embeddings = embed_model.encode(target_combined, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
        t_addr_embeddings = embed_model.encode(target_addrs, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
        np.save(t_emb_path, t_embeddings)
        np.save(t_addr_emb_path, t_addr_embeddings)
        
    if os.path.exists(s1_emb_path):
        s1_embeddings = np.load(s1_emb_path)
        s1_addr_embeddings = np.load(s1_addr_emb_path)
    else:
        s1_embeddings = embed_model.encode(s1_combined, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
        s1_addr_embeddings = embed_model.encode(s1_addrs, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
        np.save(s1_emb_path, s1_embeddings)
        np.save(s1_addr_emb_path, s1_addr_embeddings)
        
    print('Building FAISS Semantic Index...')
    faiss_index = faiss.IndexFlatIP(t_embeddings.shape[1])
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    faiss_index.add(t_embeddings)
    
    print('Retrieving Semantic Candidates...')
    top_k_faiss = 350
    faiss_distances, faiss_indices = faiss_index.search(s1_embeddings, top_k_faiss)
    faiss_cands = collections.defaultdict(list)
    for i, sid in enumerate(test_s1_ids):
        for j in range(top_k_faiss):
            tid = target_ids_list[faiss_indices[i][j]]
            score = faiss_distances[i][j]
            if score >= 0.20:
                faiss_cands[sid].append((tid, score))

    print('\n[4/5] Extracting CPU Features...')
    test_X_path = os.path.join(cache_dir, 'X_test.npy')
    test_pairs_path = os.path.join(cache_dir, 'test_pairs.pkl')
    
    if os.path.exists(test_X_path) and os.path.exists(test_pairs_path):
        print(' -> Found cached X_test! Loading from disk...')
        X_test = np.load(test_X_path)
        test_pairs = joblib.load(test_pairs_path)
    else:
        def process_test_sid(sid):
            s1_keys = s1_blocking_keys[sid]
            country = s1_preprocessed[sid][5]
            bm25_cands = get_candidates(s1_keys, country, top_k=250)
            bm25_ids = [tid for tid, _ in bm25_cands]
            f_cands = [tid for tid, _ in faiss_cands.get(sid, [])]
            cand_ids = list(set(bm25_ids + f_cands))
            
            local_X = []
            local_pairs = []
            
            s1_tup = s1_preprocessed[sid]
            s1_n_tfidf = tfidf_vec.transform([s1_tup[0]])
            s1_emb = s1_embeddings[test_s1_ids.index(sid)]
            s1_addr_emb = s1_addr_embeddings[test_s1_ids.index(sid)]
            
            for tid in cand_ids:
                t_tup = target_preprocessed[tid]
                t_n_tfidf = tfidf_vec.transform([t_tup[0]])
                tfidf_sim = (s1_n_tfidf * t_n_tfidf.T).A[0][0]
                
                t_idx = target_ids_list.index(tid)
                sem_sim = float(np.dot(s1_emb, t_embeddings[t_idx]))
                addr_sem_sim = float(np.dot(s1_addr_emb, t_addr_embeddings[t_idx]))
                
                feats = extract_features_for_pair(s1_tup, t_tup, tfidf_sim, sem_sim, addr_sem_sim, 0.0)
                local_X.append(feats)
                local_pairs.append((sid, tid, feats, s1_tup, t_tup))
                
            return local_X, local_pairs

        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(process_test_sid)(sid) for sid in tqdm(test_s1_ids, desc='Extracting')
        )
        
        X_test = []
        test_pairs = []
        for rx, rp in results:
            X_test.extend(rx)
            test_pairs.extend(rp)
            
        print('Saving extracted test features to disk...')
        X_test = np.array(X_test, dtype=np.float32)
        np.save(test_X_path, X_test)
        joblib.dump(test_pairs, test_pairs_path)

    print('\n[5/5] Scoring Pairs with GPU Cross-Encoder and XGBoost...')
    test_scores_path = os.path.join(cache_dir, 'test_cross_scores.npy')
    if os.path.exists(test_scores_path):
        print(' -> Found cached test cross-encoder scores!')
        test_cross_scores = np.load(test_scores_path)
    else:
        print('Loading ms-marco Cross-Encoder...')
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        cross_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        cross_model.eval()
        cross_model.half().to('cuda')
        
        test_cross_scores = []
        batch_size_ce = 512
        with torch.no_grad():
            for i in tqdm(range(0, len(test_pairs), batch_size_ce), desc='Batches'):
                batch = test_pairs[i:i+batch_size_ce]
                texts = [[p[3][0] + " " + p[3][2], p[4][0] + " " + p[4][2]] for p in batch]
                encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
                outputs = cross_model(**encodings)
                logits = outputs.logits.squeeze(-1).cpu().numpy().tolist()
                if isinstance(logits, float):
                    logits = [logits]
                test_cross_scores.extend(logits)
                
        test_cross_scores = np.array(test_cross_scores, dtype=np.float32)
        np.save(test_scores_path, test_cross_scores)

    for i in range(len(test_pairs)):
        prob = 1.0 / (1.0 + math.exp(-test_cross_scores[i]))
        X_test[i][-4] = float(prob)

    print('Scoring with final XGBoost model...')
    probas = final_model.predict_proba(X_test)
    
    scores_dict = collections.defaultdict(list)
    for (sid, tid, feats, s1_tup, t_tup), p in zip(test_pairs, probas):
        scores_dict[sid].append((tid, float(p)))

    print('Applying Optimal Thresholds and 1-to-1 Bipartite Matching...')
    preds = apply_threshold_and_deduplication(scores_dict, s2_threshold, s3_threshold)

    out_file = os.path.join(args.output_dir, 'submission.csv')
    print(f'Writing submission to {out_file}...')
    write_submission(preds, test_s1_ids, out_file)
    
    print(f'\nInference completed in {time.time()-t_start:.1f}s.')

if __name__ == '__main__':
    main()
