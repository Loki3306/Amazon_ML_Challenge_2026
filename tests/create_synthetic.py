import os
import pandas as pd

def create_synthetic_data():
    os.makedirs("tests/synthetic_data", exist_ok=True)
    
    # Tiny synthetic dataset for S1
    s1_data = [
        {"entity_id": "SYN-1", "business_name": "Normal Corp", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "SYN-2", "business_name": "Punctuation!!! LLC.", "business_address": "456, Broad Ave.", "country": "India"},
        {"entity_id": "SYN-3", "business_name": "Café Unicode ëxample", "business_address": "Rüe de l'München", "country": "France"},
        {"entity_id": "SYN-4", "business_name": "Missing Address Inc", "business_address": None, "country": "US"},
    ]
    
    # Tiny synthetic dataset for S2 (some exact matches)
    s2_data = [
        {"entity_id": "S2-1", "business_name": "normal corp", "business_address": "123 main st", "country": "US"}, # Matches SYN-1 name and address
        {"entity_id": "S2-2", "business_name": "Punctuation LLC", "business_address": "456 Broad Ave", "country": "India"}, # Matches SYN-2 name and address
    ]
    
    # Tiny synthetic dataset for S3 (some exact matches)
    s3_data = [
        {"entity_id": "S3-1", "business_name": "Cafe Unicode example", "business_address": "Rue de l Munchen", "country": "France"}, # Matches SYN-3 name and address
        {"entity_id": "S3-2", "business_name": "missing address inc", "business_address": "", "country": "US"}, # Matches SYN-4 name
    ]
    
    pd.DataFrame(s1_data).to_csv("tests/synthetic_data/synthetic_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_data).to_csv("tests/synthetic_data/synthetic_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_data).to_csv("tests/synthetic_data/synthetic_source3.tsv", sep="\t", index=False)
    
    # Create the config file
    config_yaml = """
train:
  source1: "tests/synthetic_data/synthetic_source1.tsv"
  source2: "tests/synthetic_data/synthetic_source2.tsv"
  source3: "tests/synthetic_data/synthetic_source3.tsv"
"""
    with open("config/dataset_synthetic.yaml", "w") as f:
        f.write(config_yaml)
        
if __name__ == "__main__":
    create_synthetic_data()
