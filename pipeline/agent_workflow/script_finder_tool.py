import os
from typing import List, Optional, Tuple

from .workflow_base import BaseScriptFinderTool
from get_mapping import get_file_mapping
from config_manager import get_config

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
        """Searches files """
        
        results = []
        
        for file_path in self.file_list:
            stripped_name = self.strip_extension(os.path.basename(file_path))
            for file_name in script_names:
                stripped_target = self.strip_extension(file_name)
                if self.mapping.get(stripped_name, "").endswith(stripped_target):
                    results.append(file_path)

        return results

    def get_description(self) -> Tuple[str, str, List[str]]:
        usage = "Read specific files. Use RARELY and only when necessary due to high token cost; "\
            "use grep_tool with sources instead whenever possible."
        parameter = "Comma-separated filenames or path fragments."
        examples = [
            "<parameter>config.nvn, utils/db.nvn</parameter>"
        ]
        
        return usage, parameter, examples

# FOR TESTING
# if __name__ == "__main__":
#     print('execution')
#     finder = PathScriptFinder(search_dirs=["./pipeline/agent_workflow/search_test_folder"])
#     finder.create_path_dict_from_txt(mapping_path="./pipeline/agent_workflow/search_test_folder/mapping_test.txt")
#     content = finder.read_file(original_path="./bonjour/bonsoir/coucou3.txt", search_dirs=finder.search_dirs)
