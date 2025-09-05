import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from utils import JobInput

# Add these network volume constants
VOLUME_PATH = "/runpod-volume"
MODELS_DIR = os.path.join(VOLUME_PATH, "models")

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    # required but ignored
    api_key='ollama',
)

# Add this NetworkVolumeManager class
class NetworkVolumeManager:
    """Manages network volume operations for Ollama models"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.setup_complete = False
        
    def setup_models_directory(self):
        """Setup Ollama to use network volume for models"""
        if self.setup_complete:
            return
            
        print(f"Setting up network volume at: {self.models_dir}")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set Ollama models directory to network volume
        os.environ["OLLAMA_MODELS"] = self.models_dir
        
        self.setup_complete = True
        print(f"Network volume setup complete. Models will be stored at: {self.models_dir}")
    
    def model_exists(self, model_name):
        """Check if a model exists on the network volume"""
        try:
            # Use ollama list to check for models
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                env=dict(os.environ, OLLAMA_MODELS=self.models_dir)
            )
            
            if result.returncode == 0:
                # Check if model_name appears in the output
                exists = model_name in result.stdout
                if exists:
                    print(f"✓ Model {model_name} found on network volume")
                else:
                    print(f"✗ Model {model_name} not found on volume, will download")
                return exists
            else:
                print(f"✗ Failed to list models: {result.stderr}")
                return False
        except Exception as e:
            print(f"✗ Error checking model existence: {e}")
            return False
    
    def ensure_model_available(self, model_name):
        """Ensure model is available, download if necessary"""
        start_time = time.time()
        
        if self.model_exists(model_name):
            load_time = time.time() - start_time
            print(f"Model {model_name} loaded from volume in {load_time:.2f}s")
            return True
        else:
            print(f"Downloading model {model_name} to network volume...")
            download_start = time.time()
            
            # Use ollama pull command with the volume-configured environment
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                env=dict(os.environ, OLLAMA_MODELS=self.models_dir)
            )
            
            if result.returncode == 0:
                download_time = time.time() - download_start
                print(f"✓ Model {model_name} downloaded successfully in {download_time:.2f}s")
                return True
            else:
                print(f"✗ Failed to download model {model_name}: {result.stderr}")
                return False

# Create global instance
volume_manager = NetworkVolumeManager()

class OllamaEngine:
    def __init__(self):
        load_dotenv()
        # Setup network volume before any Ollama operations
        volume_manager.setup_models_directory()
        print("OllamaEngine initialized")

    async def generate(self, job_input):
        # Get model from MODEL_NAME defaulting to llama3.2:1b
        model = os.getenv("MODEL_NAME", "llama3.2:1b")
        
        # ADD THIS: Ensure the model is available before proceeding
        print(f"Ensuring model {model} is available...")
        if not volume_manager.ensure_model_available(model):
            raise Exception(f"Failed to ensure model {model} is available")

        # Rest of your existing code stays exactly the same
        if isinstance(job_input.llm_input, str):
            openAiJob = JobInput({
                "openai_route": "/v1/completions",
                "openai_input": {
                    "model": model,
                    "prompt": job_input.llm_input,
                    "stream": job_input.stream
                }
            })
        else:
            openAiJob = JobInput({
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "model": model,
                    "messages": job_input.llm_input,
                    "stream": job_input.stream
                }
            })

        print("Generating response for job_input:", job_input)
        print("OpenAI job:", openAiJob)
        
        openAIEngine = OllamaOpenAiEngine()
        generate = openAIEngine.generate(openAiJob)

        async for batch in generate:
            yield batch

class OllamaOpenAiEngine(OllamaEngine):
    def __init__(self):
        load_dotenv()
        # Setup network volume before any Ollama operations  
        volume_manager.setup_models_directory()
        print("OllamaOpenAiEngine initialized")

    # Rest of your existing methods stay exactly the same
    async def generate(self, job_input):
        print("Generating response for job_input:", job_input)

        openai_input = job_input.openai_input

        if job_input.openai_route == "/v1/models":
            async for response in self._handle_model_request():
                yield response
        elif job_input.openai_route in ["/v1/chat/completions", "/v1/completions"]:
            async for response in self._handle_chat_or_completion_request(openai_input, chat=job_input.openai_route == "/v1/chat/completions"):
                yield response
        else:
            yield {"error": "Invalid route"}

    async def _handle_model_request(self):
        try:
            response = client.models.list()
            yield {"object": "list", "data": [model.to_dict() for model in response.data]} 
        except Exception as e:
            yield {"error": str(e)}

    async def _handle_chat_or_completion_request(self, openai_input, chat=False):
        try:
            if chat:
                response = client.chat.completions.create(**openai_input)
            else:
                response = client.completions.create(**openai_input)

            if not openai_input.get("stream", False):
                yield response.to_dict()
                return

            for chunk in response:
                print("Message:", chunk)
                yield "data: " + json.dumps(chunk.to_dict(), separators=(',', ':')) + "\n\n"

            yield "data: [DONE]"
        except Exception as e:
            yield {"error": str(e)}
