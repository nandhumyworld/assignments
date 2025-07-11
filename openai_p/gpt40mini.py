import os
from openai import OpenAI
from dotenv import load_dotenv

def test_gpt4o_mini():
    """
    Test basic text prompting with GPT-4o Mini model
    """
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables")
            print("Make sure your .env file contains: OPENAI_API_KEY=your_key_here")
            return
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Test prompt
        prompt = "Explain what artificial intelligence is in simple terms, in about 2-3 sentences."
        
        print("Testing GPT-4o Mini...")
        print(f"Prompt: {prompt}")
        print("-" * 50)
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        # Extract and display response
        ai_response = response.choices[0].message.content
        print("GPT-4o Mini Response:")
        print(ai_response)
        print("-" * 50)
        
        # Display usage information
        usage = response.usage
        print(f"Tokens used - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False

def interactive_chat():
    """
    Interactive chat session with GPT-4o Mini
    """
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("Error: OPENAI_API_KEY not found")
            return
        
        client = OpenAI(api_key=api_key)
        
        print("Interactive Chat with GPT-4o Mini")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                print(f"AI: {ai_response}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
    except Exception as e:
        print(f"Setup error: {str(e)}")

if __name__ == "__main__":
    print("OpenAI GPT-4o Mini Test Script")
    print("=" * 40)
    
    # Run basic test
    if test_gpt4o_mini():
        print("\nBasic test completed successfully!")
        
        # Ask if user wants to try interactive chat
        try_interactive = input("\nWould you like to try interactive chat? (y/n): ").strip().lower()
        if try_interactive in ['y', 'yes']:
            interactive_chat()
    else:
        print("\nBasic test failed. Please check your API key and connection.")