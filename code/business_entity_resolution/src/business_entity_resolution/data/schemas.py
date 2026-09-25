import polars as pl

# All columns are explicitly read as Utf8 (String) to prevent parsing errors on noisy data.
SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}

GROUND_TRUTH_SCHEMA = {
    "source1_entity_id": pl.Utf8,
    "matched_entity_ids": pl.Utf8,
}
