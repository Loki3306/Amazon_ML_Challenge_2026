import os
import pandas as pd

def create_synthetic_data():
    os.makedirs("tests/synthetic_data", exist_ok=True)
    
    # Tiny synthetic dataset based on requirements
    data = [
        {"entity_id": "SYN-1", "business_name": "Normal Corp", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "SYN-2", "business_name": "Punctuation!!! LLC.", "business_address": "456, Broad Ave.", "country": "India"},
        {"entity_id": "SYN-3", "business_name": "Café Unicode ëxample", "business_address": "Rüe de l'München", "country": "France"},
        {"entity_id": "SYN-4", "business_name": "Missing Address Inc", "business_address": None, "country": "US"},
        {"entity_id": "SYN-5", "business_name": "Empty String LLC", "business_address": "", "country": "US"},
        {"entity_id": "SYN-6", "business_name": "Duplicate Name", "business_address": "Diff Address 1", "country": "US"},
        {"entity_id": "SYN-7", "business_name": "Duplicate Name", "business_address": "Diff Address 2", "country": "India"},
        {"entity_id": "SYN-8", "business_name": "Unique Name 1", "business_address": "Duplicate Address", "country": "France"},
        {"entity_id": "SYN-9", "business_name": "Unique Name 2", "business_address": "Duplicate Address", "country": "France"},
        {"entity_id": "SYN-10", "business_name": "   Whitespace    Test  ", "business_address": "   Space   Street  ", "country": "US"},
    ]
    
    df = pd.DataFrame(data)
    
    # Save as TSV
    file_path = "tests/synthetic_data/synthetic_source1.tsv"
    df.to_csv(file_path, sep="\t", index=False)
    
    # Create the config file
    config_yaml = f"""
train:
  source1: "{file_path}"
"""
    with open("config/dataset_synthetic.yaml", "w") as f:
        f.write(config_yaml)
        
    print(f"Created synthetic dataset at {file_path}")
    print("Created synthetic config at config/dataset_synthetic.yaml")

if __name__ == "__main__":
    create_synthetic_data()
