from pathlib import Path
from typing import Dict, Any

import yaml

from get_mapping import get_file_mapping

class ConfigLoader:
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override values into the base dictionary."""
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ConfigLoader._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """
        Load env_graph configuration.

        Priority:
        1. Package defaults from env_graph/config.yaml
        2. Project-level overrides from envision/config.yaml -> env_graph section
        3. Explicit config file path if provided and it contains either full env_graph config
           or an `env_graph` subsection.
        """
        package_root = Path(__file__).resolve().parent
        package_config_path = package_root / "config.yaml"
        project_root = package_root.parent
        project_config_path = project_root / "config.yaml"

        config: Dict[str, Any] = {}

        if package_config_path.exists():
            with open(package_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        if project_config_path.exists():
            with open(project_config_path, "r", encoding="utf-8") as f:
                project_config = yaml.safe_load(f) or {}
            project_env_graph = project_config.get("env_graph")
            if isinstance(project_env_graph, dict):
                config = ConfigLoader._deep_merge(config, project_env_graph)

        explicit_path = Path(config_path)
        if config_path and explicit_path.exists() and explicit_path.resolve() != package_config_path.resolve():
            with open(explicit_path, "r", encoding="utf-8") as f:
                explicit_config = yaml.safe_load(f) or {}
            if isinstance(explicit_config.get("env_graph"), dict):
                explicit_config = explicit_config["env_graph"]
            config = ConfigLoader._deep_merge(config, explicit_config)

        return config

    @staticmethod
    def load_mapping(config: dict) -> dict:
        return get_file_mapping(config.get("mapping_file", None))

    @staticmethod
    def get_logical_path(filename: str, mapping: dict, extension: str = "nvn") -> str:
        base = filename.replace(f'.{extension}', '')
        return mapping.get(base, filename)

    @staticmethod
    def clean_path(raw_path: str) -> str:
        # Preserve placeholders like \{...\} to avoid phantom root nodes
        clean = raw_path.replace('\\', '/')
        # Collapse multiple slashes (e.g. // -> /)
        while '//' in clean:
            clean = clean.replace('//', '/')
            
        if not clean.startswith('/') and ('/' in clean or '.' in clean):
             clean = '/' + clean
        return clean
