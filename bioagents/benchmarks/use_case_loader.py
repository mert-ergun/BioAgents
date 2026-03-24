"""Loader for use-case definitions from YAML or JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from bioagents.benchmarks.use_case_models import ExperimentConfig, UseCase

logger = logging.getLogger(__name__)

# Default directory searched when no path is given
DEFAULT_USE_CASES_DIR = Path(__file__).parent.parent.parent / "use_cases"


def load_use_cases_from_file(path: str | Path) -> list[UseCase]:
    """
    Load use cases from a single YAML or JSON file.

    The file must contain either:
    - A mapping with a top-level "use_cases" key whose value is a list, or
    - A bare list of use-case dicts.

    Args:
        path: Path to .yaml/.json file.

    Returns:
        List of UseCase objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Use case file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            raw = yaml.safe_load(f)
        elif path.suffix == ".json":
            raw = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}. Use .yaml or .json")

    if isinstance(raw, dict):
        raw = raw.get("use_cases", [])

    if not isinstance(raw, list):
        logger.warning("Use case file %s did not contain a list; skipping.", path)
        return []

    use_cases: list[UseCase] = []
    for item in raw:
        try:
            use_cases.append(UseCase.from_dict(item))
        except Exception as exc:
            logger.warning("Skipping invalid use case entry in %s: %s", path, exc)

    logger.info("Loaded %d use cases from %s", len(use_cases), path)
    return use_cases


def load_use_cases_from_dir(directory: str | Path | None = None) -> list[UseCase]:
    """
    Load all use cases from every .yaml/.yml/.json file in a directory.

    Args:
        directory: Directory to scan. Defaults to the project-level ``use_cases/`` folder.

    Returns:
        Combined list of UseCase objects from all files.
    """
    directory = Path(directory) if directory else DEFAULT_USE_CASES_DIR

    if not directory.exists():
        logger.warning("Use cases directory not found: %s", directory)
        return []

    all_use_cases: list[UseCase] = []
    for path in sorted(directory.glob("**/*")):
        if path.is_file() and path.suffix in {".yaml", ".yml", ".json"}:
            try:
                all_use_cases.extend(load_use_cases_from_file(path))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

    logger.info("Loaded %d use cases total from %s", len(all_use_cases), directory)
    return all_use_cases


def load_experiment_configs_from_file(path: str | Path) -> list[ExperimentConfig]:
    """
    Load experiment configs from a single YAML or JSON file.

    The file must contain either:
    - A mapping with a "configs" or "experiment_configs" key, or
    - A bare list of config dicts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f) if path.suffix in {".yaml", ".yml"} else json.load(f)

    if isinstance(raw, dict):
        raw = raw.get("experiment_configs", raw.get("configs", []))

    if not isinstance(raw, list):
        logger.warning("Config file %s did not contain a list; skipping.", path)
        return []

    configs: list[ExperimentConfig] = []
    for item in raw:
        try:
            configs.append(ExperimentConfig.from_dict(item))
        except Exception as exc:
            logger.warning("Skipping invalid config entry in %s: %s", path, exc)

    logger.info("Loaded %d experiment configs from %s", len(configs), path)
    return configs


def filter_use_cases(
    use_cases: list[UseCase],
    ids: list[str] | None = None,
    tags: list[str] | None = None,
    category: str | None = None,
) -> list[UseCase]:
    """
    Filter a list of use cases by id, tags, or category.

    Args:
        use_cases: Full list to filter.
        ids: If provided, only return use cases whose id is in this list.
        tags: If provided, only return use cases that have at least one matching tag.
        category: If provided, only return use cases matching this category.

    Returns:
        Filtered list.
    """
    result = use_cases

    if ids is not None:
        id_set = set(ids)
        result = [uc for uc in result if uc.id in id_set]

    if tags:
        tag_set = set(tags)
        result = [uc for uc in result if tag_set.intersection(uc.tags)]

    if category is not None:
        result = [uc for uc in result if uc.category == category]

    return result
