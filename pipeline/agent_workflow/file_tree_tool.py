from typing import Dict
from get_mapping import build_file_tree
from config_manager import get_config
from rag.utils.handle_tokens import get_token_count
from pipeline.agent_workflow.concrete_workflow import BaseTreeTool
from pathlib import PurePath

class FileTreeTool(BaseTreeTool):
    def __init__(self, file_tree: Dict[str, str] = None):
        if not file_tree:
            file_tree = build_file_tree()
        
        self.tree = file_tree
    
    def render_condensed_tree(
        self,
        tree: Dict[str, str],
        indent: str = "", 
        current_depth: int = 0, 
        max_depth: int = 3, 
        max_children: int = 10
    ) -> list[str]:
        """
        Renders a nested dictionary as a tree string with condensation limits.
        
        Args:
            tree: The nested dictionary representing the file structure.
            indent: Current indentation string (used in recursion).
            current_depth: Current depth level (used in recursion).
            max_depth: How many levels deep to render before summarizing.
            max_children: How many items to show per folder before summarizing.
        """
        lines = []
        
        # 1. Check Depth Limit
        if current_depth > max_depth:
            return [f"{indent}└── ... (max depth reached)"]

        items = list(tree.items())
        # Optional: Sort so folders/files are consistent (e.g., alphabetical)
        items.sort(key=lambda x: x[0])
        
        total_items = len(items)
        
        # 2. Check Child Limit
        # If we exceed the limit, we slice the list and flag that we need a summary line
        if total_items > max_children and current_depth > 0:
            items_to_show = items[:max_children]
            remaining_count = total_items - max_children
            has_hidden_items = True
        else:
            items_to_show = items
            remaining_count = 0
            has_hidden_items = False

        for i, (name, subtree) in enumerate(items_to_show):
            # Determine if this is visually the last item in this branch
            # It is the last if: 
            # a) It's the last in the full list AND we aren't hiding any items
            # b) We are hiding items, so this is NOT the last (the summary line is last)
            is_last_visual = (i == len(items_to_show) - 1) and not has_hidden_items
            
            prefix = "└── " if is_last_visual else "├── "
            lines.append(f"{indent}{prefix}{name}")
            
            # Recurse if it is a directory (subtree is a dict and not empty)
            if isinstance(subtree, dict) and subtree:
                lines[-1] += "/"
                extension = indent + ("    " if is_last_visual else "│   ")
                lines.extend(
                    self.render_condensed_tree(
                        subtree, 
                        extension, 
                        current_depth + 1, 
                        max_depth, 
                        max_children
                    )
                )

        # 3. Add Summary Line if needed
        if has_hidden_items:
            lines.append(f"{indent}└── ... ({remaining_count} more items)")

        return lines

    def fit_tree_to_context(
        self,
        tree_root: dict, 
        max_tokens: int = 1000
    ) -> str:
        """
        Automatically adjusts tree rendering parameters to fit within a token limit.
        
        Args:
            tree_root: The nested dictionary of the file tree.
            max_tokens: The maximum number of tokens allowed for the summary.
            token_estimator_func: Function to count tokens (defaults to char/4).
            
        Returns:
            The rendered tree string that fits the limit.
        """
        
        # Define "Detail Levels" from most detailed to least detailed.
        # Format: (max_depth, max_children)
        # We try these sequentially until one fits.
        backoff_levels = get_config().get("main_pipeline.tree_backoff_levels")
        
        last_rendered_tree = ""
        
        for depth, children in backoff_levels:
            # Generate the tree with current settings
            lines = self.render_condensed_tree(
                tree_root, 
                max_depth=depth, 
                max_children=children
            )
            rendered_tree = "\n".join(lines)
            
            # Check token count
            count = get_token_count(rendered_tree)
            
            # Store this result in case even the strictest level fails 
            # (we return the smallest one we managed to generate)
            last_rendered_tree = rendered_tree
            
            if count <= max_tokens:
                return rendered_tree
                
        # If we run out of levels and it's still too big, return the smallest version
        # optionally with a warning prepended.
        return last_rendered_tree
    
    def tree_tool(self, root_path: str, max_tokens: int = 1000) -> str:
        """Returns string render of the file tree starting from the root path.
        If the root_path is invalid it returns the tree from the larges prefix path
        which is valid.

        Args:
            root_path (str): the root path to render the tree from. E.g. "/1. utilities/Modules"
            max_tokens (int, optional): The maximum number of tokens allowed in the output. Defaults to 1000.

        Returns:
            str: The rendered file tree string.
        """
        path_obj = PurePath(root_path)
        parts = [p for p in path_obj.parts if p and p != path_obj.anchor]
        tree_root = self.tree
        res = "/"
        for part in parts:
            if part in tree_root:
                tree_root = tree_root[part]
                res += f"{part}/"
            else:
                break
        
        res += "\n"
        
        return res + self.fit_tree_to_context(tree_root, max_tokens)

if __name__ == "__main__":
    tool = FileTreeTool()
    tree_str = tool.tree_tool("", max_tokens=500)
    print(tree_str)