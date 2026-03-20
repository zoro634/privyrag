import os

def pull_models():
    print("Pulling required models with Ollama...")
    print("This might take a while depending on your internet connection.")
    # Embedding model
    print("Pulling nomic-embed-text...")
    os.system("ollama pull nomic-embed-text")
    # Generation model
    print("Pulling llama3...")
    os.system("ollama pull llama3")
    print("Models pulled successfully.")

if __name__ == "__main__":
    pull_models()
