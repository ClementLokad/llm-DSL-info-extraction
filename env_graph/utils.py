import yaml
from pathlib import Path
from typing import Dict, Any
from get_mapping import get_file_mapping

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """Loads YAML config relative to the package root or absolute path."""
        # Priority 1: Package default config
        package_root = Path(__file__).parent
        path = package_root / "config.yaml"
        
        # Priority 2: Override from user provided path (if exists)
        if Path(config_path).exists() and config_path != "config.yaml":
             path = Path(config_path)

        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

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
