"""Load config/default.yaml and the subject list into one dataclass."""
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


@dataclass
class Config:
    """All settings, grouped the same way as the YAML file."""

    seed: int
    paths: dict
    data: dict
    vocab: dict
    model: dict
    train: dict
    generate: dict
    filters: dict
    paper: dict
    eval: dict
    subjects: dict
    marks: dict

    def path(self, name: str) -> Path:
        """Return an absolute path for a key under `paths`."""
        return ROOT / self.paths[name]


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    """Read the main YAML and merge in subjects_topics.yaml."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    topics = yaml.safe_load((ROOT / raw["paths"]["subjects_topics"]).read_text(encoding="utf-8"))
    return Config(**raw, subjects=topics["subjects"], marks=topics["marks"])
