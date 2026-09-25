import polars as pl
from .dataset_config import get_path
from .schemas import SOURCE_SCHEMA, GROUND_TRUTH_SCHEMA

def load_source1(split="train", lazy=False):
    path = get_path(split, "source1")
    return _load_tsv(path, SOURCE_SCHEMA, lazy=lazy)

def load_source2(split="train", lazy=False):
    path = get_path(split, "source2")
    return _load_tsv(path, SOURCE_SCHEMA, lazy=lazy)

def load_source3(split="train", lazy=False):
    path = get_path(split, "source3")
    return _load_tsv(path, SOURCE_SCHEMA, lazy=lazy)

def load_ground_truth(split="train", lazy=False):
    path = get_path(split, "ground_truth")
    return _load_tsv(path, GROUND_TRUTH_SCHEMA, lazy=lazy)

def _load_tsv(path, schema, lazy=False):
    if lazy:
        return pl.scan_csv(path, separator="\t", dtypes=schema, infer_schema_length=0, null_values=[""])
    return pl.read_csv(path, separator="\t", dtypes=schema, infer_schema_length=0, null_values=[""])
