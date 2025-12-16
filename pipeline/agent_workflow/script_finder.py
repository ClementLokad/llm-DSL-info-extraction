from typing import List, Dict, Any, Optional

from .workflow_base import BaseScriptFinderTool

class PathScriptFinder(BaseScriptFinderTool):
    """Script finder that searches for scripts in a the filesystem made of the folders listed in search dirs."""
    
    def __init__(self, search_dirs: List[str]):
        self.search_dirs = search_dirs
        self.tree_conversion_tree = None

    def create_path_dict_from_txt(self, mapping_path: str) -> Dict[str, str]:
        """Create a dictionary linking before and after scrape name and path using a txt mapping containing file_name, original_path on each line"""
        conversion_dict = {}
        with open(mapping_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                file_name, original_path = line.strip().split(', ')
                conversion_dict[original_path] = file_name 
        self.tree_conversion_dict = conversion_dict

    def read_file(self, original_path: str, search_dirs: List[str]) -> Optional[str]:
        """Searches a file based on its original path in the specified search directories and returns its content. """
        import glob
        import os

        file_name = self.tree_conversion_dict[original_path]
        search_dirs = self.search_dirs
        found_file_path = None

        # Pattern without extension to allow search by name only
        generic_file_pattern = f"{file_name}.*"

        # Search for the file in the specified directories
        for directory in search_dirs:
            normalized_directory = os.path.normpath(directory) # Ensures the format of the path is correct for the OS
            search_pattern = os.path.join(normalized_directory, generic_file_pattern)
            found_files = glob.glob(search_pattern)

            # Exit after first instance is found
            if found_files:
                found_file_path = found_files[0]
                print(f"File found in '{directory}' : {found_file_path}")
                break

        if found_file_path:
            try:
                with open(found_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print("\n--- READING FILE CONTENT ---")
                    preview = content[:200] + ('...' if len(content) > 200 else '')
                    print(preview)
                    print("--- END OF FILE CONTENT ---\n")
                    return content
                
            except Exception as e:
                print(f"Couldn't read the following file {found_file_path}: {e}")
                return None
            
        else:
            print(f"File '{file_name}' not found.")
            return None

# FOR TESTING
# if __name__ == "__main__":
#     print('execution')
#     finder = PathScriptFinder(search_dirs=["./pipeline/agent_workflow/search_test_folder"])
#     finder.create_path_dict_from_txt(mapping_path="./pipeline/agent_workflow/search_test_folder/mapping_test.txt")
#     content = finder.read_file(original_path="./bonjour/bonsoir/coucou3.txt", search_dirs=finder.search_dirs)
