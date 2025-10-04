#!/usr/bin/env python3
"""
Script to list available Gemini models and test connectivity.
"""

import os
import sys
from config_manager import get_config

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """List available Gemini models"""
    print("🔍 Checking Gemini API connectivity and available models...")
    print()
    
    # Check if API key is set
    config = get_config()
    api_key = config.get_api_key('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("💡 Make sure to set your API key in the .env file")
        return
    
    print(f"✅ API key found: {api_key[:10]}...")
    print()
    
    try:
        from agents.gemini_agent import GeminiAgent
        
        # List available models
        available_models = GeminiAgent.list_available_models()
        
        if available_models:
            print(f"📊 Found {len(available_models)} available models")
            print()
            print("🧪 Testing connectivity with the first available model...")
            
            # Test with the first available model
            test_model = available_models[0]
            agent = GeminiAgent(model=test_model)
            agent.initialize()
            
            response = agent.generate_response("Say hello in one word")
            print(f"✅ Test successful with model '{test_model}'")
            print(f"📝 Response: {response}")
            print()
            print(f"💡 Recommended usage:")
            print(f"   agent = GeminiAgent(model='{test_model}')")
            
        else:
            print("❌ No models available or API connection failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print()
        print("💡 Make sure:")
        print("   1. Your GOOGLE_API_KEY is valid")
        print("   2. You have access to Gemini API")
        print("   3. You've installed required dependencies: pip install google-generativeai")


if __name__ == "__main__":
    main()