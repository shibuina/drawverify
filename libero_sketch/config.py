"""Config loading: YAML file -> nested dict with attribute access + CLI overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None = None, overrides: dict | None = None) -> Cfg:
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, overrides)
    cfg = Cfg(copy.deepcopy(cfg))
    return cfg


def out_root(cfg: Cfg, cli_value: str | None = None) -> Path:
    root = Path(cli_value or cfg.paths.out_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def data_root(cfg: Cfg, cli_value: str | None = None) -> Path:
    """LIBERO datasets folder. CLI > config > $LIBERO_DATASETS.
    Relative paths resolve against data_gen/ (not the CWD)."""
    v = cli_value or cfg.paths.libero_datasets or os.environ.get("LIBERO_DATASETS")
    if not v:
        raise SystemExit(
            "LIBERO dataset root not set: pass --data-root, set paths.libero_datasets "
            "in the config, or export LIBERO_DATASETS")
    p = Path(v).expanduser()
    if not p.is_absolute():
        p = DEFAULT_CONFIG_PATH.parent.parent / p
    return p
