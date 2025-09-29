import os
from dotenv import load_dotenv
from agents.gpt_agent import GPTAgent
from agents.mistral_agent import MistralAgent
from agents.gemini_agent import GeminiAgent

load_dotenv()

MODEL_NAME = 'gemini'

def initialize_agent():
    """
    Initialize the agent based on the selected MODEL_NAME.
    """
    try:
        # Create the appropriate agent
        agent = None
        if MODEL_NAME.lower() == 'gpt':
            agent = GPTAgent()
        elif MODEL_NAME.lower() == 'mistral':
            agent = MistralAgent()
        elif MODEL_NAME.lower() == 'gemini':
            agent = GeminiAgent()
        else:
            raise ValueError(f"Model '{MODEL_NAME}' not supported. Use 'gpt', 'mistral' or 'gemini'.")
        
        # Initialize the agent
        agent.initialize()
        return agent
            
    except (ValueError, RuntimeError) as e:
        print(f"\nInitialization error: {str(e)}")
        print("\nMake sure API keys are defined in the .env file:")
        print("- OPENAI_API_KEY for GPT")
        print("- MISTRAL_API_KEY for Mistral")
        print("- GOOGLE_API_KEY for Gemini")
        raise

def main():
    # Initialize the agent
    try:
        agent = initialize_agent()
        print(f"\nAgent {MODEL_NAME.upper()} initialized successfully!")
        print("Type 'quit' or 'exit' to quit.\n")
        
        # Chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            try:
                # Send question to agent and get response
                response = agent.process_question(user_input)
                print(f"\n{MODEL_NAME.upper()}: {response}")
                
            except Exception as e:
                print(f"\nError processing question: {str(e)}")
    
    except Exception as e:
        print(f"Initialization error: {str(e)}")

if __name__ == "__main__":
    main()