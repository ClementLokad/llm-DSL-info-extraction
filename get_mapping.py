import os
from typing import Dict
from config_manager import get_config

def get_file_mapping(mapping_file_path: str = None) -> Dict[str, str]:
    """
    Reads a mapping file that associates file IDs with their original file paths.
    """
    if mapping_file_path is None:
        mapping_file_path = get_config().get('paths.mapping_path')
    # 1. Parse the mapping file into a dictionary
    # Structure: {'48623': '/users/dev/script.py', ...}
    file_mapping = {}
    
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split only on the first comma to avoid issues if the path itself contains commas
                parts = line.split(',', 1)
                
                if len(parts) == 2:
                    file_id = parts[0].strip()
                    original_path = parts[1].strip()
                    
                    # Store in dictionary
                    file_mapping[file_id] = original_path
                else:
                    print(f"Skipping malformed line in mapping: {line}")
    except FileNotFoundError:
        print(f"Error: Could not find mapping file at {mapping_file_path}")
        return
    
    return file_mapping

def get_inverse_mapping(mapping_file_path: str = None) -> Dict[str, str]:
    """
    Reads a mapping file and returns the inverse mapping from original file paths to file IDs.
    """
    original_mapping = get_file_mapping(mapping_file_path)
    
    # Invert the mapping
    inverse_mapping = {v: k for k, v in original_mapping.items()}
    return inverse_mapping
