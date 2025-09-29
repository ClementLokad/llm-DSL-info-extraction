"""
Test the preprocessing configuration and setup.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from preprocessing.config import registry

def test_preprocessing_setup():
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file")
        return False
        
    # Configure Gemini
    try:
        genai.configure(api_key=api_key)
        # List available models
        for m in genai.list_models():
            print(f"Found model: {m.name}")
        
        model = genai.GenerativeModel('models/gemini-2.5-pro')
        # Quick test of the model
        response = model.generate_content("Test connection")
        print("✓ Gemini API connection successful")
    except Exception as e:
        print(f"Error connecting to Gemini API: {e}")
        return False
    
    # Test model registry
    try:
        # Get Gemini configuration
        config = registry.get_config('gemini', 'models/gemini-2.5-pro')
        print(f"\nActive model configuration:")
        print(f"- Model type: {config.model_type}")
        print(f"- Model name: {config.name}")
        print(f"- Embedding dimension: {config.embedding_dim}")
        print(f"- Context length: {config.context_length}")
        
        # List all available models (showing architecture supports multiple models)
        print("\nAvailable models in registry:")
        for model_type, models in registry.list_available_models().items():
            print(f"- {model_type}: {', '.join(models)}")
            
        registry.set_active_config('gemini', 'models/gemini-2.5-pro')
        print("\n✓ Model registry configuration successful")
    except Exception as e:
        print(f"Error testing model registry: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("Testing preprocessing configuration and API setup...")
    test_preprocessing_setup()