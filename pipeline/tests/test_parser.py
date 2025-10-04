"""
Test script for the Envision parser.

This script tests the parsing capabilities on sample Envision DSL files.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocessing.parsers.envision_parser import EnvisionParser
from preprocessing.utils.helpers import setup_logging

def test_envision_parser():
    """Test the Envision parser on sample files."""
    setup_logging(level="INFO")
    
    parser = EnvisionParser()
    
    # Test file path
    test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "env_scripts", "67982.nvn")
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    print(f"Testing Envision parser on: {test_file}")
    print("=" * 60)
    
    try:
        # Parse the file
        blocks = parser.parse_file(test_file)
        
        print(f"Successfully parsed {len(blocks)} code blocks:")
        print()
        
        # Group blocks by type
        block_types = {}
        for block in blocks:
            block_type = block.block_type
            if block_type not in block_types:
                block_types[block_type] = []
            block_types[block_type].append(block)
        
        # Display summary
        for block_type, type_blocks in block_types.items():
            print(f"{block_type.replace('_', ' ').title()}: {len(type_blocks)} blocks")
            
            # Show first few examples
            for i, block in enumerate(type_blocks[:3]):
                name = block.name or f"Block_{block.line_start}"
                content_preview = block.content[:100].replace('\n', ' ')
                print(f"  {i+1}. {name} (lines {block.line_start}-{block.line_end})")
                print(f"     Preview: {content_preview}...")
                
                # Show metadata
                if block.metadata:
                    metadata_items = []
                    for key, value in block.metadata.items():
                        if key != 'section':  # Section is obvious
                            metadata_items.append(f"{key}: {value}")
                    if metadata_items:
                        print(f"     Metadata: {', '.join(metadata_items)}")
                print(f"    Source: {os.path.basename(block.file_path)}\n")
            
            if len(type_blocks) > 3:
                print(f"  ... and {len(type_blocks) - 3} more")
                print()
        
        # Test dependency extraction
        print("Dependency Analysis:")
        print("-" * 30)
        
        dependencies = {}
        for block in blocks:
            if block.name:
                deps = parser.extract_dependencies(block)
                if deps:
                    dependencies[block.name] = deps
        
        for name, deps in list(dependencies.items())[:5]:  # Show first 5
            print(f"{name}: {', '.join(deps)}")
        
        if len(dependencies) > 5:
            print(f"... and {len(dependencies) - 5} more blocks with dependencies")
        
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_envision_parser()