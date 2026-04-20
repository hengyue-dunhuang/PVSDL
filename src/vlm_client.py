"""
VLM Client Module
Wraps OpenRouter API calls and supports multimodal (image + text) inputs.
"""
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.utils import encode_image_to_base64


class VLMClient:
    def __init__(self, api_key=None, base_url="https://openrouter.ai/api/v1"):
        """
        Initialize the VLM Client.
        
        Args:
            api_key: OpenRouter API key. Loads from environment variables if None.
            base_url: Base URL for the API.
        """
        # Load environment variables
        load_dotenv()
        
        # Retrieve API Key
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError("OPENROUTER_API_KEY not found. Please set it in your .env file or pass it as an argument.")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Load model configurations
        self.model_config = self._load_model_config()
        
        print(f"✓ VLM Client initialized successfully")
    
    def _load_model_config(self):
        """Load model configuration from local JSON file."""
        try:
            with open('config/models.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Map model names to their respective configurations
                model_map = {}
                for model in config['models']:
                    model_map[model['name']] = model
                return model_map
        except Exception as e:
            print(f"Warning: Failed to load model configuration: {e}")
            return {}
    
    def _supports_reasoning(self, model_name):
        """
        Check if the model supports reasoning/CoT features.
        
        Args:
            model_name: The name/ID of the model.
            
        Returns:
            bool: True if reasoning is supported.
        """
        # Search in loaded config
        if model_name in self.model_config:
            return self.model_config[model_name].get('supports_reasoning', False)
        
        # Default to True; the API usually ignores this if unsupported by the specific model
        return True
    
    def query_image(self, image_path, prompt, model_name, max_retries=3):
        """
        Query the VLM with an image and a text prompt.
        
        Args:
            image_path: Path to the image file.
            prompt: Text instruction for the model.
            model_name: Model ID (e.g., openai/gpt-4o).
            max_retries: Maximum number of retry attempts.
            
        Returns:
            dict: Dictionary containing response content and metadata.
        """
        # Encode image to base64
        try:
            image_base64 = encode_image_to_base64(image_path)
        except Exception as e:
            return {
                "success": False,
                "error": f"Image encoding failed: {str(e)}",
                "response": None
            }
        
        # Construct message payload
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        # Check if reasoning features should be enabled
        enable_reasoning = self._supports_reasoning(model_name)
        
        # Retry mechanism
        for attempt in range(max_retries):
            try:
                # Build API parameters
                api_params = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 5000  # Limited as we typically expect classification results
                }
                
                # Add extra body if reasoning is supported
                if enable_reasoning:
                    api_params["extra_body"] = {"reasoning": {"enabled": True}}
                
                # Call the API
                response = self.client.chat.completions.create(**api_params)
                
                # Extract message content
                message = response.choices[0].message
                content = message.content
                
                # Extract reasoning details if provided by the model
                reasoning_details = None
                if hasattr(message, 'reasoning_details') and message.reasoning_details:
                    reasoning_details = message.reasoning_details
                
                return {
                    "success": True,
                    "response": content,
                    "model": model_name,
                    "reasoning_details": reasoning_details,
                    "reasoning_enabled": enable_reasoning,
                    "usage": {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', None),
                        "total_tokens": getattr(response.usage, 'total_tokens', None)
                    },
                    "error": None
                }
                
            except Exception as e:
                error_msg = str(e)
                print(f"  Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # All attempts exhausted
                    return {
                        "success": False,
                        "error": error_msg,
                        "response": None
                    }
    
    def batch_query(self, image_list, prompt, model_name, verbose=True):
        """
        Query the VLM for multiple images in a batch.
        
        Args:
            image_list: List of image info dictionaries (containing 'path', 'filename', etc.).
            prompt: Text instruction for the model.
            model_name: Model ID.
            verbose: Whether to print detailed progress logs.
            
        Returns:
            list: List of query results.
        """
        results = []
        total = len(image_list)
        
        if verbose:
            print(f"\nStarting batch query: {total} images")
            print(f"Model: {model_name}")
            print(f"Prompt: {prompt[:50]}...")
        
        for idx, image_info in enumerate(image_list, 1):
            if verbose:
                print(f"\n[{idx}/{total}] Processing: {image_info['filename']}")
            
            # Query single image
            result = self.query_image(
                image_path=image_info['path'],
                prompt=prompt,
                model_name=model_name
            )
            
            # Append metadata
            result['image_info'] = image_info
            result['index'] = idx
            
            results.append(result)
            
            if verbose and result['success']:
                print(f"  Response: {result['response'][:100]}")
                # Show reasoning logs
                if result.get('reasoning_enabled'):
                    if result.get('reasoning_details'):
                        print(f"  Reasoning Details: {str(result['reasoning_details'])[:100]}...")
                    else:
                        print(f"  Reasoning Mode: Enabled (No details provided)")
            elif verbose:
                print(f"  Error: {result['error']}")
            
            # Slight delay to respect API rate limits
            if idx < total:
                time.sleep(0.5)
        
        if verbose:
            success_count = sum(1 for r in results if r['success'])
            print(f"\n✓ Batch query finished: {success_count}/{total} successful")
        
        return results


if __name__ == "__main__":
    # Test Suite
    print("Testing VLM Client...")
    
    try:
        client = VLMClient()
        
        # Test single image query
        test_image = "dataset/archive/solar_panel_dust_segmentation/images/Imgclean_0_0.jpg"
        test_prompt = "Is this solar panel clean or dirty? Answer with only 'clean' or 'dirty'."
        
        print(f"\nTesting single image query:")
        print(f"Image: {test_image}")
        print(f"Prompt: {test_prompt}")
        
        result = client.query_image(
            image_path=test_image,
            prompt=test_prompt,
            model_name="moonshotai/kimi-k2.5"
        )
        
        if result['success']:
            print(f"\n✓ Query Successful!")
            print(f"Response: {result['response']}")
            print(f"Token Usage: {result['usage']}")
        else:
            print(f"\n✗ Query Failed: {result['error']}")
            
    except Exception as e:
        print(f"Test failed: {e}")