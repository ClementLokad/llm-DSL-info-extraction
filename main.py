"""
Main entry point for the LLM DSL Info Extraction project.

This project aims to build an AI assistant for LOKAD's Supply Chain Scientists
to understand and navigate complex Envision DSL codebases.

Usage:
    python main.py --help
"""

import argparse
import sys
from pathlib import Path
from test import run_tests, interactive_mode, create_agent

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LLM DSL Info Extraction - AI Assistant for Envision Code Analysis"
    )
    
    parser.add_argument(
        "--model", 
        choices=["gpt", "mistral", "gemini"],
        default="gemini",
        help="LLM model to use (default: gemini)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["test", "interactive"],
        default="interactive", 
        help="Run mode: test or interactive (default: interactive)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="LLM DSL Info Extraction v0.1.0"
    )
    
    args = parser.parse_args()
    
    print("🚀 LLM DSL Info Extraction")
    print("=" * 50)
    print(f"Model: {args.model.upper()}")
    print(f"Mode: {args.mode}")
    print()
    
    # Set global model for test module
    import test
    test.MODEL_NAME = args.model
    
    try:
        if args.mode == "test":
            print("Running test suite...")
            success = run_tests()
            sys.exit(0 if success else 1)
        else:
            print("Starting interactive mode...")
            interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()