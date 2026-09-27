"""
Feature Engineering Module
High-impact pairwise features combining TF-IDF cosine score and SIMD string metrics.
"""

import numpy as np
from rapidfuzz import fuzz, distance

FEATURE_NAMES = [
    "tfidf_score",       # Blocking TF-IDF cosine similarity
    "jw_name",           # Jaro-Winkler similarity on normalized name
    "tsort_name",        # Token Sort Ratio on name
    "tset_name",         # Token Set Ratio on name
    "tsort_addr",        # Token Sort Ratio on address
    "postal_match",      # Exact postal match (1.0), prefix match (0.5), or mismatch/missing (0.0)
    "len_ratio_name"     # Ratio of shorter to longer name
]


def extract_pair_features(
    sn: str,
    sa: str,
    sp: str,
    cn: str,
    ca: str,
    cp: str,
    tfidf_score: float
) -> list:
    """
    Extract the 7 most decisive features for a single candidate pair.
    """
    f_tfidf = float(tfidf_score)
    f_jw = distance.JaroWinkler.similarity(sn, cn)
    f_tsort_n = fuzz.token_sort_ratio(sn, cn) / 100.0
    f_tset_n = fuzz.token_set_ratio(sn, cn) / 100.0
    
    if sa and ca:
        f_tsort_a = fuzz.token_sort_ratio(sa, ca) / 100.0
    else:
        f_tsort_a = 0.0
        
    if sp and cp:
        if sp == cp:
            f_postal = 1.0
        elif len(sp) >= 3 and len(cp) >= 3 and sp[:3] == cp[:3]:
            f_postal = 0.5
        else:
            f_postal = 0.0
    else:
        f_postal = 0.0
        
    l1, l2 = len(sn), len(cn)
    f_len_ratio = min(l1, l2) / max(l1, l2, 1)
    
    return [f_tfidf, f_jw, f_tsort_n, f_tset_n, f_tsort_a, f_postal, f_len_ratio]


def extract_features_batch(
    cand_dict: dict,
    s1_lookup: dict,
    cand_lookup: dict
) -> tuple:
    """
    Extract features for all candidate pairs in batches.
    """
    X_list = []
    pair_keys = []
    
    for sid, ctuples in cand_dict.items():
        if sid not in s1_lookup:
            continue
        sn, sa, sp = s1_lookup[sid]
        
        for cid, score in ctuples:
            if cid in cand_lookup:
                cn, ca, cp = cand_lookup[cid]
                feats = extract_pair_features(sn, sa, sp, cn, ca, cp, score)
                X_list.append(feats)
                pair_keys.append((sid, cid))
                
    if not X_list:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []
        
    return np.array(X_list, dtype=np.float32), pair_keys
