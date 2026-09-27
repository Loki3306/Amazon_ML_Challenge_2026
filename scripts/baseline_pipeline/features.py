import sys
import math
import numpy as np
import re
import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
import normalization as norm

FEATURE_NAMES = [
    'name_exact_clean',
    'name_exact_core',
    'name_lev_sim',
    'name_jw_sim',
    'name_token_sort',
    'name_token_set',
    'name_token_jaccard',
    'name_token_overlap',
    'name_len_diff',
    'name_len_ratio',
    'name_prefix_match',
    'name_char_3gram',
    'name_first_token_match',
    'name_first_token_conflict',
    'addr_has_val',
    'addr_exact_clean',
    'addr_lev_sim',
    'addr_jw_sim',
    'addr_token_sort',
    'addr_token_set',
    'addr_token_jaccard',
    'num_common_count',
    'num_has_match',
    'num_has_mismatch',
    'primary_num_match',
    'primary_num_mismatch',
    'primary_num_diff',
    'exact_core_no_addr',
    'cross_prod',
    'cross_sum',
    'cross_max',
    'cross_min',
    'name_high_but_num_conflict',
    'is_source2',
    'is_source3',
    'shared_keys_count',
    'name_phonetic_match',
    'addr_phonetic_match',
    'zip_code_match',
    'name_tfidf_sim',
    'name_semantic_sim',
    'addr_semantic_sim',
    'cross_encoder_score',
    'is_addr_missing',
    'is_name_missing',
    'is_pnum_missing'
]


def char_ngrams(s, n=3):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def extract_features_for_pair(s1_tuple, target_tuple, target_id, shared_keys=1, tfidf_sim=0.0, semantic_sim=0.0, addr_semantic_sim=0.0, cross_score=0.0):
    """
    s1_tuple: (clean_name, core_name, clean_addr, nums_set, primary_num)
    target_tuple: (clean_name, core_name, clean_addr, nums_set, primary_num)
    target_id: string (e.g. S2-12345 or S3-67890)
    """
    s1_name, s1_core, s1_addr, s1_nums, s1_pnum = s1_tuple
    t_name, t_core, t_addr, t_nums, t_pnum = target_tuple
    
    is_addr_missing = 1.0 if not s1_addr or not t_addr else 0.0
    is_name_missing = 1.0 if not s1_name or not t_name else 0.0
    is_pnum_missing = 1.0 if s1_pnum is None or t_pnum is None else 0.0

    # 1. Name features
    exact_clean = 1.0 if s1_name == t_name and s1_name else 0.0
    exact_core = 1.0 if s1_core == t_core and s1_core else 0.0

    s1_phon = jellyfish.metaphone(s1_name) if s1_name else ""
    t_phon = jellyfish.metaphone(t_name) if t_name else ""
    name_phonetic_match = 1.0 if s1_phon and t_phon and s1_phon == t_phon else 0.0

    n_lev = Levenshtein.normalized_similarity(s1_name, t_name) if s1_name and t_name else 0.0
    n_jw = JaroWinkler.similarity(s1_name, t_name) if s1_name and t_name else 0.0
    n_sort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0
    n_set = fuzz.token_set_ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0

    s1_toks = s1_core.split()
    t_toks = t_core.split()
    if s1_toks and t_toks:
        s1_s = set(s1_toks)
        t_s = set(t_toks)
        intersection = len(s1_s & t_s)
        union = len(s1_s | t_s)
        n_jaccard = intersection / union if union > 0 else 0.0
        n_overlap = intersection / min(len(s1_s), len(t_s))

        # First token match / conflict on significant tokens
        f1_sim = Levenshtein.normalized_similarity(s1_toks[0], t_toks[0])
        first_token_match = 1.0 if f1_sim >= 0.80 else 0.0
        first_token_conflict = 1.0 if f1_sim < 0.50 else 0.0
    else:
        n_jaccard = 0.0
        n_overlap = 0.0
        first_token_match = 0.0
        first_token_conflict = 0.0

    len1 = len(s1_name)
    len2 = len(t_name)
    n_len_diff = abs(len1 - len2)
    n_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
    prefix_match = 1.0 if len(s1_core) >= 4 and len(t_core) >= 4 and s1_core[:4] == t_core[:4] else 0.0

    # Character 3-gram similarity on core name
    ng1 = char_ngrams(s1_core, 3)
    ng2 = char_ngrams(t_core, 3)
    if ng1 and ng2:
        char_3gram_sim = len(ng1 & ng2) / len(ng1 | ng2)
    else:
        char_3gram_sim = 0.0

    # 2. Address features
    has_addr = 1.0 if t_addr else 0.0
    exact_core_no_addr = 1.0 if (not has_addr and exact_core == 1.0) else 0.0

    if has_addr and s1_addr:
        a_exact = 1.0 if s1_addr == t_addr else 0.0
        a_lev = Levenshtein.normalized_similarity(s1_addr, t_addr)
        a_jw = JaroWinkler.similarity(s1_addr, t_addr)
        a_sort = fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0
        a_set = fuzz.token_set_ratio(s1_addr, t_addr) / 100.0

        s1_a_toks = set(s1_addr.split())
        t_a_toks = set(t_addr.split())
        a_union = len(s1_a_toks | t_a_toks)
        a_jaccard = len(s1_a_toks & t_a_toks) / a_union if a_union > 0 else 0.0

        s1_a_phon = jellyfish.metaphone(s1_addr) if s1_addr else ""
        t_a_phon = jellyfish.metaphone(t_addr) if t_addr else ""
        addr_phonetic_match = 1.0 if s1_a_phon and t_a_phon and s1_a_phon == t_a_phon else 0.0

        zip_pattern = r'\b(\d{5,6})\b'
        s1_zips = set(re.findall(zip_pattern, s1_addr))
        t_zips = set(re.findall(zip_pattern, t_addr))
        if s1_zips and t_zips:
            zip_code_match = 1.0 if len(s1_zips & t_zips) > 0 else -1.0
        else:
            zip_code_match = 0.0

        # Number overlap
        shared_nums = len(s1_nums & t_nums)
        has_num_match = 1.0 if shared_nums > 0 else 0.0
        has_num_mismatch = 1.0 if len(s1_nums) > 0 and len(t_nums) > 0 and shared_nums == 0 else 0.0

        # Primary street number comparison
        if s1_pnum is not None and t_pnum is not None:
            prim_match = 1.0 if s1_pnum == t_pnum else 0.0
            prim_mismatch = 1.0 if s1_pnum != t_pnum else 0.0
            try:
                diff_val = math.log1p(abs(int(s1_pnum) - int(t_pnum)))
            except Exception:
                diff_val = 0.0 if s1_pnum == t_pnum else 1.0
        else:
            prim_match = 0.0
            prim_mismatch = 0.0
            diff_val = 0.0
    else:
        a_exact = 0.0
        a_lev = 0.0
        a_jw = 0.0
        a_sort = 0.0
        a_set = 0.0
        a_jaccard = 0.0
        addr_phonetic_match = 0.0
        zip_code_match = 0.0
        shared_nums = 0
        has_num_match = 0.0
        has_num_mismatch = 0.0
        prim_match = 0.0
        prim_mismatch = 0.0
        diff_val = 0.0

    # 3. Cross-field interaction features
    effective_addr_sim = a_sort if has_addr else n_sort
    cross_prod = n_sort * effective_addr_sim
    cross_sum = n_sort + effective_addr_sim
    cross_max = max(n_sort, effective_addr_sim)
    cross_min = min(n_sort, effective_addr_sim)

    # Flag look-alikes on same street with conflicting primary numbers
    name_high_but_num_conflict = 1.0 if (n_sort >= 0.75 and prim_mismatch == 1.0) else 0.0

    # 4. Source indicators
    is_s2 = 1.0 if target_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if target_id.startswith('S3-') else 0.0

    return [
        exact_clean,
        exact_core,
        n_lev,
        n_jw,
        n_sort,
        n_set,
        n_jaccard,
        n_overlap,
        n_len_diff,
        n_len_ratio,
        prefix_match,
        char_3gram_sim,
        first_token_match,
        first_token_conflict,
        has_addr,
        a_exact,
        a_lev,
        a_jw,
        a_sort,
        a_set,
        a_jaccard,
        float(shared_nums),
        has_num_match,
        has_num_mismatch,
        prim_match,
        prim_mismatch,
        float(diff_val),
        exact_core_no_addr,
        cross_prod,
        cross_sum,
        cross_max,
        cross_min,
        name_high_but_num_conflict,
        is_s2,
        is_s3,
        float(shared_keys),
        name_phonetic_match,
        addr_phonetic_match,
        zip_code_match,
        float(tfidf_sim),
        float(semantic_sim),
        float(addr_semantic_sim),
        float(cross_score),
        is_addr_missing,
        is_name_missing,
        is_pnum_missing
    ]
