import os
import re
from typing import List, Optional, Dict, Any

from .workflow_base import BaseScriptFinderTool, _tool_desc
from utils.get_mapping import get_file_mapping
from utils.config_manager import get_config

class PathScriptFinder(BaseScriptFinderTool):
    """Script finder that searches for scripts in a the filesystem made of the folders listed in search dirs."""
    
    def __init__(self, search_dirs: Optional[List[str]] = None):
        if search_dirs is None:
            search_dirs = get_config().get('paths.input_dirs', [])
        super().__init__(search_dirs)
        self.mapping = get_file_mapping()
        self.file_list = self._get_all_files(self.search_dirs)
        
    def _get_all_files(self, root_dirs: List[str]) -> List[str]:
        """
        Recursively gets the full path of all files in a directory and its subdirectories.

        Args:
            root_dir (str): The starting directory path.

        Returns:
            list: A list of absolute file paths.
        """
        all_files : set = set()
        for root_dir in root_dirs:
            # os.walk is the most efficient way to traverse a directory tree
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # 'dirpath' is the current directory we are in
                # 'filenames' is a list of all files in the current 'dirpath'

                for filename in filenames:
                    # os.path.join creates a full, correct path string
                    full_path = os.path.join(dirpath, filename)
                    all_files.add(full_path)

        return list(all_files)

    def strip_extension(self, file_path):
        """Helper to safely remove the file extension from a path."""
        # os.path.splitext returns a tuple (root, ext)
        root, ext = os.path.splitext(file_path)
        return root

    def find_scripts(self, script_names: List[str]) -> List[str]:
        """Searches files and returns absolute paths"""
        
        results = []
        
        for file_path in self.file_list:
            stripped_name = self.strip_extension(os.path.basename(file_path))
            for file_name in script_names:
                stripped_target = str(self.strip_extension(file_name)).strip("/")
                original_path = str(self.strip_extension(self.mapping.get(stripped_name, "")))
                if re.search(stripped_target, original_path, re.IGNORECASE):
                    results.append(file_path)

        return results
    
    def original_path(self, path: str) -> str:
        stripped_name = self.strip_extension(os.path.basename(path))
        if stripped_name in self.mapping:
            return self.mapping[stripped_name]
        else:
            return "Unknown Path"

    def get_description(self) -> Dict[str, Any]:
        return _tool_desc(
            name="script_finder_tool",
            description=(
                "Read specific files in full. Use RARELY and only when necessary due to high token cost; "
                "use grep_tool with a sources filter instead whenever possible."
            ),
            properties={
                "script_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of filenames or path fragments to locate and read. "
                        "E.g. ['Functions.nvn', '/7. Documentation/1 Project']."
                    ),
                },
            },
            required=["script_names"],
        )

# FOR TESTING
if __name__ == "__main__":
    print('execution')
    finder = PathScriptFinder()
    res = finder.find_scripts(["1 - Item Inspector"])
    print([finder.original_path(r) for r in res])
