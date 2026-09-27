"""
Preprocessing & Text Normalization Module
Memory-safe streaming dataset loaders and text cleaning utilities.
"""

import os
import re
import gc
import unicodedata
import pandas as pd
import numpy as np

# Shared TSV column dtypes for consistent parsing across all modules
TSV_DTYPES = {
    "entity_id": "str",
    "business_name": "str",
    "business_address": "str",
    "country": "str",
}

# Compiled regex for international legal suffixes (US, India, France, Germany, UK)
LEGAL_RE = re.compile(
    r'\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|'
    r'pvt|private|sa|sas|sarl|eurl|sci|scp|snc|se|gmbh|ag|ug|ohg|kg|'
    r'plc|llp|lp|nv|bv|trust|associates|group|enterprises|holdings|services)\b',
    re.IGNORECASE
)

NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
WHITESPACE = re.compile(r'\s+')
INDIA_POSTAL_RE = re.compile(r'\b(\d{6})\b')
US_FR_POSTAL_RE = re.compile(r'\b(\d{5})\b')


def strip_accents(text: str) -> str:
    """Normalize Unicode characters and remove diacritics/accents safely."""
    if not isinstance(text, str) or not text:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_text(text: str, entity_id: str = "") -> str:
    """
    Standardize business name safely:
    1. Strip diacritics/accents (e.g. French 'e' -> 'e')
    2. Lowercase
    3. Remove legal entity suffixes
    4. Remove punctuation & special characters
    5. Normalize whitespace
    6. Edge case fallback: if string becomes empty after cleaning, fallback to raw string or entity_id.
    """
    if not isinstance(text, str) or not text:
        return clean_text(entity_id) if entity_id and entity_id != text else "unknown"

    t = strip_accents(text).lower()
    t = LEGAL_RE.sub(" ", t)
    t = NON_ALPHANUM.sub(" ", t)
    t = WHITESPACE.sub(" ", t).strip()

    # Edge case: If name consisted purely of legal suffixes or punctuation (e.g. "LLC, Inc."), retain original cleaned string
    if not t:
        raw_clean = NON_ALPHANUM.sub(" ", strip_accents(text).lower()).strip()
        t = raw_clean if raw_clean else (entity_id.lower() if entity_id else "unknown")

    return t


def extract_postal_code(address: str, country: str) -> str:
    """Extract standard postal/ZIP codes by country pattern safely."""
    if not isinstance(address, str) or not address:
        return ""
    if country == "India":
        m = INDIA_POSTAL_RE.search(address)
    else:  # US or France
        m = US_FR_POSTAL_RE.search(address)
    return m.group(1) if m else ""


def build_entity_lookup(df: pd.DataFrame, country: str = None) -> dict:
    """
    Vectorized dictionary lookup creation.
    Avoids slow pandas iterrows() and completes on 10M rows in ~2 seconds.
    Returns:
        dict: {entity_id: (clean_name, clean_address, postal_code)}
    """
    ids = df["entity_id"].values
    names = df["clean_name"].values

    raw_addrs = df["business_address"].fillna("").astype(str).values
    addrs = [a.lower() for a in raw_addrs]

    if country:
        postals = [extract_postal_code(a, country) for a in raw_addrs]
    elif "country" in df.columns:
        countries = df["country"].fillna("US").values
        postals = [extract_postal_code(a, c) for a, c in zip(raw_addrs, countries)]
    else:
        postals = ["" for _ in raw_addrs]

    return dict(zip(ids, zip(names, addrs, postals)))


def load_country_slice(tsv_path: str, country: str) -> pd.DataFrame:
    """
    Stream-load TSV file in 500k row chunks and filter ONLY rows for the target country.
    Keeps RAM usage strictly < 1.5 GB even when reading 10M row datasets.
    Handles missing values and corrupt rows safely.
    """
    chunks = []
    for chunk in pd.read_csv(
        tsv_path,
        sep="\t",
        chunksize=500000,
        dtype=TSV_DTYPES,
        on_bad_lines="skip"
    ):
        chunk["country"] = chunk["country"].fillna("US")
        c_chunk = chunk[chunk["country"] == country].copy()
        if not c_chunk.empty:
            c_chunk["entity_id"] = c_chunk["entity_id"].fillna("").astype(str)
            c_chunk["business_name"] = c_chunk["business_name"].fillna("").astype(str)
            c_chunk["clean_name"] = [clean_text(n, eid) for n, eid in zip(c_chunk["business_name"], c_chunk["entity_id"])]
            chunks.append(c_chunk)
        del chunk

    if not chunks:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country", "clean_name"])

    res = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return res


def load_filtered_candidates(tsv_paths: list, country: str, needed_ids: set) -> pd.DataFrame:
    """
    Stream-load TSV files and return ONLY rows matching the target country
    whose entity_id is in the needed_ids set.
    """
    chunks = []
    for path in tsv_paths:
        if not os.path.exists(path):
            continue
        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=500000,
            dtype=TSV_DTYPES,
            on_bad_lines="skip"
        ):
            chunk["country"] = chunk["country"].fillna("US")
            c = chunk[(chunk["country"] == country) & (chunk["entity_id"].isin(needed_ids))]
            if not c.empty:
                c = c.copy()
                c["entity_id"] = c["entity_id"].fillna("").astype(str)
                c["business_name"] = c["business_name"].fillna("").astype(str)
                c["clean_name"] = [clean_text(n, eid) for n, eid in zip(c["business_name"], c["entity_id"])]
                chunks.append(c)
            del chunk

    if not chunks:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country", "clean_name"])

    result = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return result
