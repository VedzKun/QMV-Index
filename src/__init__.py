"""
QMV Index — source package.
Import config loader here so every module can do:
    from src import load_config
"""
import yaml
from pathlib import Path


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load and return the master YAML config as a nested dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)
