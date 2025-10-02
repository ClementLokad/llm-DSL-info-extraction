import os
from dotenv import load_dotenv
from agents.gpt_agent import GPTAgent
from agents.mistral_agent import MistralAgent
from agents.gemini_agent import GeminiAgent

load_dotenv()

# Configuration - Change this to test different models
MODEL_NAME = 'gemini'


def create_agent(model_name: str, specific_model: str = None):
    """
    Create and initialize an agent based on the model name.
    
    Args:
        model_name: The model to use ('gpt', 'mistral', or 'gemini')
        specific_model: Specific model variant (for Gemini: 'gemini-2.5-flash', etc.)
        
    Returns:
        Initialized LLM agent
    """
    model_name = model_name.lower()
    
    if model_name == 'gpt':
        agent = GPTAgent()
    elif model_name == 'mistral':
        agent = MistralAgent()
    elif model_name == 'gemini':
        if specific_model:
            agent = GeminiAgent(model=specific_model)
        else:
            agent = GeminiAgent()
    else:
        raise ValueError(f"Model '{model_name}' not supported. Use 'gpt', 'mistral' or 'gemini'.")
    
    # Initialize the agent
    agent.initialize()
    return agent


def test_gemini_models():
    """Test different Gemini models to find one that works"""
    print("\n🔍 Testing different Gemini models...")
    
    # List of stable models to try
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash", 
        "gemini-pro-latest",
        "gemini-flash-latest",
        "gemini-2.5-pro"
    ]
    
    for model in models_to_try:
        print(f"\n🧪 Trying model: {model}")
        try:
            agent = GeminiAgent(model=model)
            agent.initialize()
            response = agent.generate_response("Say hello")
            print(f"✅ SUCCESS with {model}")
            print(f"📝 Response: {response}")
            return agent
        except Exception as e:
            if "quota" in str(e).lower():
                print(f"❌ Quota exceeded for {model}")
            elif "not found" in str(e).lower():
                print(f"❌ Model {model} not found")
            else:
                print(f"❌ Error with {model}: {str(e)[:100]}...")
            continue
    
    print("❌ No working Gemini models found")
    return None


def test_simple_question(agent, question: str):
    """Test a simple question without context"""
    print(f"\n🤖 Testing: {question}")
    try:
        response = agent.generate_response(question)
        print(f"✅ {agent.model_name}: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_with_context(agent, question: str, context: str):
    """Test a question with context"""
    print(f"\n🔍 Testing with context: {question}")
    try:
        response = agent.generate_response(question, context)
        print(f"✅ {agent.model_name}: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def run_tests():
    """Run basic tests on the selected model"""
    try:
        print(f"\n🚀 Initializing {MODEL_NAME.upper()} agent...")
        
        # Special handling for Gemini quota issues
        if MODEL_NAME.lower() == 'gemini':
            agent = test_gemini_models()
            if not agent:
                print("\n💡 Tips for Gemini API:")
                print("   1. Check your quota at https://ai.google.dev/gemini-api/docs/rate-limits")
                print("   2. Try again later (quotas reset daily)")
                print("   3. Try a different model using:")
                print("      agent = GeminiAgent(model='gemini-2.0-flash')")
                return False
        else:
            agent = create_agent(MODEL_NAME)
            
        print(f"✅ {agent.model_name} initialized successfully!")
        
        # Basic tests
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Simple question
        total_tests += 1
        if test_simple_question(agent, "What is Python?"):
            tests_passed += 1
            
        # Test 2: Question with context
        total_tests += 1
        context = "Python is a high-level programming language known for its simplicity."
        if test_with_context(agent, "What makes this language popular?", context):
            tests_passed += 1
        
        # Test 3: Empty question (should fail gracefully)
        total_tests += 1
        print(f"\n⚠️  Testing empty question (should fail gracefully)")
        try:
            agent.generate_response("")
            print("❌ Should have failed with empty question")
        except ValueError:
            print("✅ Correctly handled empty question")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
        
        print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
        return tests_passed == total_tests
        
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        print("\n💡 Make sure API keys are defined in the .env file:")
        print("   - OPENAI_API_KEY for GPT")
        print("   - MISTRAL_API_KEY for Mistral") 
        print("   - GOOGLE_API_KEY for Gemini")
        return False


def interactive_mode():
    """Interactive chat mode"""
    try:
        agent = create_agent(MODEL_NAME)
        print(f"\n💬 {agent.model_name} is ready! Type 'quit' or 'exit' to quit.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            if not user_input:
                print("Please enter a question.")
                continue
            
            try:
                response = agent.generate_response(user_input)
                print(f"\n{agent.model_name}: {response}\n")
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    print("🧪 LLM Agent Test Suite")
    print("=" * 50)
    
    mode = input("\nChoose mode:\n1. Run tests\n2. Interactive chat\nChoice [i]: ").strip().lower()
    
    if mode in ['t', 'test', '1']:
        success = run_tests()
        exit(0 if success else 1)
    else:
        interactive_mode()