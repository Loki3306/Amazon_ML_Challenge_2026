"""
Post-Processing & Graph Consistency Module
Greedy Maximum Weighted Bipartite Matching for cardinality resolution, singleton protection, and TSV exports.
"""

from collections import defaultdict
import numpy as np


def apply_graph_postprocessing(
    pair_records: list,
    tau_match: float,
    tau_singleton: float = 0.30,
    max_matches: int = 11
) -> dict:
    """
    Apply global graph consistency constraints:
    1. Greedy Maximum Weighted Bipartite Matching: S2/S3 entities map to at most 1 S1 entity.
    2. Singleton Protection: S1 entities with max probability < tau_singleton are protected as empty.
    3. Match Count Cap: Truncates match list to at most 11 items.
    
    Args:
        pair_records: list of ((s1_id, cand_id), probability)
        tau_match: Optimal decision threshold from validation tuning
        tau_singleton: Confidence threshold below which entity is treated as singleton
        max_matches: Maximum allowed matches per entity (11 per dataset specification)
        
    Returns:
        dict: {s1_id: set(matched_cand_ids)}
    """
    # Sort pairs descending by probability
    pair_records.sort(key=lambda x: x[1], reverse=True)
    
    assigned_cand = set()
    final_matches = defaultdict(set)
    s1_max_prob = defaultdict(float)
    pair_prob_lookup = {}
    
    # 1. Greedy Bipartite Matching (1-to-1 Cardinality Enforcement)
    for (sid, cid), prob in pair_records:
        pair_prob_lookup[(sid, cid)] = prob
        s1_max_prob[sid] = max(s1_max_prob[sid], prob)
        if prob >= tau_match and cid not in assigned_cand:
            final_matches[sid].add(cid)
            assigned_cand.add(cid)
            
    # 2. Singleton Protection
    for sid, max_p in s1_max_prob.items():
        if max_p < tau_singleton:
            final_matches[sid] = set()
            
    # 3. Match Count Cap (<= 11) using O(1) probability lookups
    for sid in list(final_matches.keys()):
        if len(final_matches[sid]) > max_matches:
            top_matches = sorted(
                final_matches[sid],
                key=lambda c: pair_prob_lookup.get((sid, c), 0.0),
                reverse=True
            )
            final_matches[sid] = set(top_matches[:max_matches])
            
    return final_matches


def write_submission_tsv(filepath: str, all_s1_ids: list, matches_dict: dict):
    """
    Write matching_results.tsv in the exact required tab-separated format:
    source1_entity_id\\tmatched_entity_ids (comma-separated, sorted, no spaces)
    Guarantees all S1 test IDs appear in exact input order.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            matches = matches_dict.get(sid, set())
            if not matches:
                f.write(f"{sid}\t\n")
            else:
                f.write(f"{sid}\t{','.join(sorted(matches))}\n")


def write_candidates_tsv(filepath: str, all_s1_ids: list, candidate_dict: dict):
    """
    Write candidate_pairs.tsv in the exact required tab-separated format.
    Guarantees all S1 test IDs appear in exact input order.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands = candidate_dict.get(sid, [])
            cand_set = sorted(set(cands))
            f.write(f"{sid}\t{','.join(cand_set)}\n")
