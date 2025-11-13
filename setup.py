import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    """Download spaCy model"""
    try:
        import spacy
        spacy.cli.download("en_core_web_sm")
        print("spaCy model downloaded successfully")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")

def create_env_file():
    """Create .env file template"""
    env_content = """# Free LLMs
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_MODEL=llama2

DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=your_deepseek_key_here
DEEPSEEK_MODEL=deepseek-chat

GROQ_BASE_URL=https://api.groq.com
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama3-70b-8192

MISTRAL_BASE_URL=https://api.mistral.ai
MISTRAL_API_KEY=your_mistral_key_here
MISTRAL_MODEL=mistral-medium

HF_BASE_URL=https://api-inference.huggingface.co
HF_API_KEY=your_hf_key_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.1

# Paid LLMs
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-pro

OPENAI_BASE_URL=https://api.openai.com
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4

ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

PERPLEXITY_BASE_URL=https://api.perplexity.ai
PERPLEXITY_API_KEY=your_perplexity_key_here
PERPLEXITY_MODEL=sonar-medium-online
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print(".env file created. Please update with your API keys.")

if __name__ == "__main__":
    print("Setting up Brand Visibility SuperAgent...")
    
    # Install requirements
    print("Installing requirements...")
    install_requirements()
    
    # Download spaCy model
    print("Downloading spaCy model...")
    download_spacy_model()
    
    # Create .env file
    print("Creating .env file...")
    create_env_file()
    
    print("\nSetup complete!")
    print("1. Update the .env file with your API keys")
    print("2. Run: python main.py")
    print("3. Follow the prompts to analyze brand visibility")
