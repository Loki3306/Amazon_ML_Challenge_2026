import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "dataset.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()

def get_path(split: str, source: str) -> str:
    if split not in config:
        raise ValueError(f"Unknown split: {split}")
    if source not in config[split]:
        raise ValueError(f"Unknown source: {source} in split {split}")
    
    # Paths are relative to project root. We assume this code is run from project root.
    return config[split][source]
