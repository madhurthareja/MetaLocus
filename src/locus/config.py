"""Config loading: config/default.yaml, overridable by LOCUS_<SECTION>__<KEY>
env vars, validated on load.

`device: auto` resolves to cuda if available else cpu -- this is the one
place allowed to ask "is a GPU here", so geometry/memory/eval code never
has to.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

REQUIRED_SECTIONS = {"perception", "memory", "kinematics", "hazard", "image"}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    prefix = "LOCUS_"
    for env_key, value in os.environ.items():
        if not env_key.startswith(prefix) or "__" not in env_key:
            continue
        section, key = env_key[len(prefix):].lower().split("__", 1)
        if section in config and isinstance(config[section], dict):
            config[section][key] = yaml.safe_load(value)
    return config


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config = _apply_env_overrides(config)

    missing = REQUIRED_SECTIONS - config.keys()
    if missing:
        raise ValueError(f"config at {config_path} is missing required sections: {missing}")

    config["device"] = resolve_device(config.get("device", "auto"))
    return config
