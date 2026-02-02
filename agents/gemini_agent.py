import google.generativeai as genai
from typing import Optional
from .base import LLMAgent, rate_limited
from config_manager import get_config


class GeminiAgent(LLMAgent):
    """Implementation of an agent using Google's Gemini
    
    Recommended stable Gemini models (as of October 2025):
    - "gemini-2.5-flash"         # Fast, stable multimodal model (default)
    - "gemini-2.5-pro"           # Most capable stable model
    - "gemini-2.0-flash"         # Alternative fast model
    - "gemini-pro-latest"        # Latest stable pro model
    - "gemini-flash-latest"      # Latest stable flash model
    
    Note: Model availability changes frequently. Use list_available_models() 
    to see current available models for your API key.
    
    Usage:
        # List available models
        GeminiAgent.list_available_models()
        
        # Use specific model
        agent = GeminiAgent(model="gemini-2.5-flash")
    """
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        """Initialize the Gemini agent
        
        Args:
            model: The Gemini model to use (see class docstring for available models)
        """
        super().__init__()
        self.api_key = None
        self.model = None
        self._model_name = model
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1000,
        }

    @classmethod
    def list_available_models(cls, api_key: Optional[str] = None):
        """
        List all available Gemini models.
        
        Args:
            api_key: Optional API key. If not provided, uses GOOGLE_API_KEY from environment.
            
        Returns:
            List of available model names
        """
        if not api_key:
            config = get_config()
            api_key = config.get_api_key('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        try:
            genai.configure(api_key=api_key)
            models = list(genai.list_models())
            
            print("🤖 Available Gemini Models:")
            print("=" * 50)
            
            for model in models:
                # Filter for generative models
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    print(f"✅ {model_name}")
                    if hasattr(model, 'description') and model.description:
                        print(f"   Description: {model.description}")
                    print()
                    
            return [model.name.replace('models/', '') for model in models 
                   if 'generateContent' in model.supported_generation_methods]
                   
        except Exception as e:
            print(f"❌ Error listing models: {str(e)}")
            return []

    def initialize(self) -> None:
        """Initialize the connection to the Gemini API"""
        config = get_config()
        self.api_key = config.get_api_key('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self._model_name)
            # Test connection with a simple prompt
            test_response = self.model.generate_content("Hello")
            if not test_response.text:
                raise Exception("Model returned empty response")
        except Exception as e:
            # If the model fails, try to suggest available models
            print(f"❌ Failed to initialize model '{self._model_name}'")
            print("🔍 Trying to list available models...")
            try:
                available_models = self.list_available_models(self.api_key)
                if available_models:
                    print(f"💡 Try using one of these models instead:")
                    print(f"   agent = GeminiAgent(model='{available_models[0]}')")
            except:
                pass
            raise ConnectionError(f"Failed to connect to Gemini API: {str(e)}")
    
    @rate_limited()
    @LLMAgent.count_tokens
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """Generate a response using Gemini"""
        if not question.strip():
            raise ValueError("Question cannot be empty")
            
        if not self.model:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        try:
            prompt = question
            if context:
                prompt = f"Given this context:\n\n{context}\n\nQuestion: {question}"
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            if not response.text:
                raise Exception("Empty response received from API")
                
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                raise Exception("Gemini API quota exceeded. Please try again later.")
            elif "rate" in error_msg.lower():
                raise Exception("Gemini API rate limit reached. Please try again in a few seconds.")
            else:
                raise Exception(f"Error calling Gemini API: {error_msg}")
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return f"Gemini-{self._model_name}"